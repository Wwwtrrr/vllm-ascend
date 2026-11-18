# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in, batched input preparation captured inside target FULL graphs."""

import torch
from vllm.triton_utils import tl, triton


def make_target_prefix(runner):
    """Reject layouts whose metadata construction reads/copies device inputs."""
    config = runner.speculative_config
    if config is None or not getattr(config, "leagle_merge_target_input_graph", False):
        return None
    from vllm_ascend.attention.attention_v1 import AscendAttentionBackend
    from vllm_ascend.worker.v2.model_states.default import AscendModelState

    parallel = runner.parallel_config
    state = getattr(runner.speculator, "target_input_graph_state", None)
    if (
        config.method != "leagle"
        or not config.leagle_target_input_graphs
        or getattr(config, "parallel_drafting", False)
        or type(runner.model_state) is not AscendModelState
        or parallel.pipeline_parallel_size != 1
        or parallel.decode_context_parallel_size != 1
        or parallel.prefill_context_parallel_size != 1
        or runner.uses_inputs_embeds
        or runner.vllm_config.kv_transfer_config is not None
        or runner.model_config.rswa_window is not None
        or getattr(runner.model_state, "kvpp_runtime", None) is not None
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
def _prepare_requests(
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
    SEQ_CAPACITY: tl.constexpr,
    NUM_TOKENS_PADDED: tl.constexpr,
    ROW_CAPACITY: tl.constexpr,
    DRAFT_STRIDE: tl.constexpr,
    BLOCK_STRIDE: tl.constexpr,
    OUTPUT_BLOCK_STRIDE: tl.constexpr,
    KV_BLOCK_SIZE: tl.constexpr,
    OUTPUT_SIZE: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
    TOKEN_TILE: tl.constexpr,
    TILE: tl.constexpr,
):
    # Also captured in target graphs used by prefill/dummy/fallback execution.
    # Disabled means no reads of live request state and no output writes.
    num_reqs = tl.load(enabled)
    if num_reqs == 0:
        return
    program = tl.program_id(0)
    block_programs = ROW_CAPACITY * BLOCK_TILES
    if program < block_programs:
        row = program // BLOCK_TILES
        offset = (program % BLOCK_TILES) * TILE + tl.arange(0, TILE)
        real = row < num_reqs
        req = tl.load(idx + row, mask=real, other=0)
        block_count = tl.load(block_counts + req, mask=real, other=0)
        block = tl.load(
            source_blocks + req * BLOCK_STRIDE + offset,
            mask=real & (offset < BLOCK_STRIDE) & (offset < block_count),
            other=0,
        )
        tl.store(blocks_out + row * OUTPUT_BLOCK_STRIDE + offset, block, mask=offset < BLOCK_STRIDE)
        return

    if program < block_programs + ROW_CAPACITY:
        row = program - block_programs
        if row < num_reqs:
            req = tl.load(idx + row)
            start = tl.load(query + row)
            end = tl.load(query + row + 1)
            logits_start = tl.load(cu + row)
            logits_end = tl.load(cu + row + 1)
            count = logits_end - logits_start
            base = tl.load(computed + req)
            tl.store(seq_lens + row, base + end - start)
            offset = tl.arange(0, TOKEN_TILE)
            valid = offset < count
            # Decode-only validation guarantees query length == logits count.
            token_index = start + offset
            pos = base + offset
            block = tl.load(source_blocks + req * BLOCK_STRIDE + pos // KV_BLOCK_SIZE, mask=valid, other=0)
            slot = block * KV_BLOCK_SIZE + pos % KV_BLOCK_SIZE
            tl.store(slots_out + token_index, slot, mask=valid)
            token = tl.load(draft + req * DRAFT_STRIDE + offset - 1, mask=valid & (offset > 0), other=0)
            anchor = tl.load(sampled + req)
            tl.store(input_ids + token_index, tl.where(offset == 0, anchor, token), mask=valid)
            tl.store(positions + token_index, pos, mask=valid)
            output_index = logits_start + offset
            output_valid = valid & (output_index < OUTPUT_SIZE)
            tl.store(expanded + output_index, req, mask=output_valid)
            tl.store(local + output_index, offset, mask=output_valid)
            tl.store(logits + output_index, token_index, mask=output_valid)
        return

    # Padding and per-request programs write disjoint ranges, so they can
    # execute in any order without a barrier. Clean the full slot capacity,
    # but initialize only the token padding consumed by this target graph.
    offset = (program - block_programs - ROW_CAPACITY) * TILE + tl.arange(0, TILE)
    actual_tokens = tl.load(query + num_reqs)
    q = tl.load(query + offset, mask=offset < QUERY_SIZE, other=0)
    tl.store(query_out + offset, q, mask=offset < QUERY_SIZE)
    tl.store(seq_lens + offset, 0, mask=(offset >= num_reqs) & (offset < SEQ_CAPACITY))
    tl.store(slots_out + offset, -1, mask=(offset >= actual_tokens) & (offset < TOKEN_CAPACITY))
    token_padding = (offset >= actual_tokens) & (offset < NUM_TOKENS_PADDED)
    tl.store(input_ids + offset, 0, mask=token_padding)
    tl.store(positions + offset, 0, mask=token_padding)


class LEagleTargetPrefix:
    """Fixed buffers shared with the ordinary target graph and its sampler."""

    def __init__(self, state):
        self.state = state
        device = state.metadata.device
        self.enabled = torch.zeros(1, dtype=torch.int32, device=device)
        self._enabled = 0
        self.captured = False
        self.pending = 0
        self.replays = 0
        self.expanded = torch.empty(state.max_requests * state.query_len, dtype=torch.int32, device=device)
        self.local = torch.empty_like(self.expanded)
        self.logits = torch.empty_like(self.expanded, dtype=torch.int64)

    def set_enabled(self, num_reqs):
        if not 0 <= num_reqs <= self.state.max_requests:
            raise ValueError("LEAGLE merged graph request count exceeds capacity")
        if self._enabled != num_reqs:
            self.enabled.fill_(num_reqs)
            self._enabled = num_reqs

    def outputs(self, count):
        if not 0 <= count <= self.expanded.numel():
            raise ValueError("LEAGLE merged graph logits count exceeds capacity")
        return self.expanded[:count], self.local[:count], self.logits[:count]

    def forward(self, num_tokens_padded):
        state = self.state
        buffers, requests, tables = state.buffers, state.requests, state.block_tables
        tile_size = 256
        # Each real query consumes at least one token. FIA may append one
        # empty padding request, so reserve that row even for an exact bucket.
        row_capacity = min(state.max_requests, num_tokens_padded + 1)
        block_stride = tables.block_tables[0].gpu.stride(0)
        block_tiles = triton.cdiv(block_stride, tile_size)
        padding_tiles = triton.cdiv(
            max(buffers.input_ids.numel(), buffers.seq_lens.numel(), state.query.numel()), tile_size
        )
        programs = row_capacity * block_tiles + row_capacity + padding_tiles
        _prepare_requests[(programs,)](
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
            SEQ_CAPACITY=buffers.seq_lens.numel(),
            NUM_TOKENS_PADDED=num_tokens_padded,
            ROW_CAPACITY=row_capacity,
            DRAFT_STRIDE=requests.draft_tokens.stride(0),
            BLOCK_STRIDE=block_stride,
            OUTPUT_BLOCK_STRIDE=tables.input_block_tables[0].stride(0),
            KV_BLOCK_SIZE=tables.kernel_block_sizes[0],
            OUTPUT_SIZE=self.expanded.numel(),
            BLOCK_TILES=block_tiles,
            TOKEN_TILE=triton.next_power_of_2(state.query_len),
            TILE=tile_size,
        )
