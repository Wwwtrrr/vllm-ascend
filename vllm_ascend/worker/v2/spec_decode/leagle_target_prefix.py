# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in, single-request input preparation captured inside target FULL graphs."""

import torch
from vllm.triton_utils import tl, triton


def make_target_prefix(runner):
    """Reject layouts whose metadata construction reads/copies device inputs."""
    config = runner.speculative_config
    if config is None or not getattr(config, "leagle_merge_target_input_graph", False):
        return None
    from vllm_ascend.attention.attention_v1 import AscendAttentionBackend

    parallel = runner.parallel_config
    state = getattr(runner.speculator, "target_input_graph_state", None)
    if (
        config.method != "leagle"
        or not config.leagle_target_input_graphs
        or runner.max_num_reqs != 1
        or runner.model_config.architecture != "Qwen3ForCausalLM"
        or parallel.tensor_parallel_size != 1
        or parallel.pipeline_parallel_size != 1
        or parallel.data_parallel_size != 1
        or parallel.decode_context_parallel_size != 1
        or parallel.prefill_context_parallel_size != 1
        or runner.lora_config is not None
        or runner.pcp_manager is not None
        or runner.adaptive_verification is not None
        or runner.model_state.rope_state is not None
        or runner.model_state.supports_mm_inputs
        or getattr(runner.model_state, "prompt_embeds_state", None) is not None
        or hasattr(runner.model_state.model, "prepare_engram_inputs")
        or getattr(runner.model_state, "_offload_req_ids", None) is not None
        or state is None
        or state.block_tables is None
        or state.block_tables.num_kv_cache_groups != 1
        or state.block_tables.cp_size != 1
        or not state.block_tables._slot_mapping_enabled[0]
        or any(group.backend is not AscendAttentionBackend for groups in runner.attn_groups for group in groups)
    ):
        from vllm.logger import logger

        logger.warning("LEAGLE merged target input graph unsupported by this layout; using separate preparation.")
        return None
    return LEagleTargetPrefix(state)


@triton.jit
def _prepare_single_request(
    enabled,
    idx,
    cu,
    query,
    query_out,
    computed,
    sampled,
    draft,
    input_ids,
    positions,
    seq_lens,
    source_blocks,
    block_counts,
    blocks_out,
    slots_out,
    expanded,
    local,
    logits,
    QUERY_SIZE: tl.constexpr,
    TOKEN_CAPACITY: tl.constexpr,
    DRAFT_STRIDE: tl.constexpr,
    BLOCK_STRIDE: tl.constexpr,
    KV_BLOCK_SIZE: tl.constexpr,
    OUTPUT_SIZE: tl.constexpr,
    TILE: tl.constexpr,
):
    # Also captured in target graphs used by prefill/dummy/fallback execution.
    # Disabled means no reads of live request state and no output writes.
    if tl.load(enabled) == 0:
        return
    req = tl.load(idx)
    count = tl.load(cu + 1)
    base = tl.load(computed + req)
    block_count = tl.load(block_counts + req)
    lane = tl.arange(0, TILE)
    q = tl.load(query + lane, mask=lane < QUERY_SIZE, other=0)
    tl.store(query_out + lane, q, mask=lane < QUERY_SIZE)
    tl.store(seq_lens, base + count)
    for start in range(0, BLOCK_STRIDE, TILE):
        offset = start + lane
        block = tl.load(
            source_blocks + req * BLOCK_STRIDE + offset, mask=(offset < BLOCK_STRIDE) & (offset < block_count), other=0
        )
        tl.store(blocks_out + offset, block, mask=offset < BLOCK_STRIDE)
    for start in range(0, TOKEN_CAPACITY, TILE):
        offset = start + lane
        valid = offset < count
        pos = base + offset
        block_idx = pos // KV_BLOCK_SIZE
        block = tl.load(source_blocks + req * BLOCK_STRIDE + block_idx, mask=valid, other=0)
        slot = block * KV_BLOCK_SIZE + pos % KV_BLOCK_SIZE
        tl.store(slots_out + offset, tl.where(valid, slot, -1), mask=offset < TOKEN_CAPACITY)
        token = tl.load(draft + req * DRAFT_STRIDE + offset - 1, mask=valid & (offset > 0), other=0)
        anchor = tl.load(sampled + req)
        tl.store(input_ids + offset, tl.where(offset == 0, anchor, token), mask=valid)
        tl.store(positions + offset, pos, mask=valid)
        tl.store(expanded + offset, req, mask=valid & (offset < OUTPUT_SIZE))
        tl.store(local + offset, offset, mask=valid & (offset < OUTPUT_SIZE))
        tl.store(logits + offset, offset, mask=valid & (offset < OUTPUT_SIZE))


class LEagleTargetPrefix:
    """Fixed buffers shared with the ordinary target graph and its sampler."""

    def __init__(self, state):
        self.state = state
        device = state.metadata.device
        self.enabled = torch.zeros(1, dtype=torch.int32, device=device)
        self._enabled = False
        self.captured = False
        self.pending = False
        self.replays = 0
        self.expanded = torch.empty(state.query_len, dtype=torch.int32, device=device)
        self.local = torch.empty_like(self.expanded)
        self.logits = torch.empty(state.query_len, dtype=torch.int64, device=device)

    def set_enabled(self, enabled):
        if self._enabled != enabled:
            self.enabled.fill_(int(enabled))
            self._enabled = enabled

    def outputs(self, count):
        return self.expanded[:count], self.local[:count], self.logits[:count]

    def forward(self):
        state = self.state
        buffers, requests, tables = state.buffers, state.requests, state.block_tables
        _prepare_single_request[(1,)](
            self.enabled,
            state.idx,
            state.cu,
            state.query,
            buffers.query_start_loc,
            requests.num_computed_tokens.gpu,
            requests.last_sampled_tokens,
            requests.draft_tokens,
            buffers.input_ids,
            buffers.positions,
            buffers.seq_lens,
            tables.block_tables[0].gpu,
            state.block_counts,
            tables.input_block_tables[0],
            tables.slot_mappings,
            self.expanded,
            self.local,
            self.logits,
            QUERY_SIZE=state.query.numel(),
            TOKEN_CAPACITY=buffers.input_ids.numel(),
            DRAFT_STRIDE=requests.draft_tokens.stride(0),
            BLOCK_STRIDE=tables.block_tables[0].gpu.stride(0),
            KV_BLOCK_SIZE=tables.kernel_block_sizes[0],
            OUTPUT_SIZE=state.query_len,
            TILE=256,
        )
