# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native preparation graphs for LEAGLE target verification inputs."""

import numpy as np
import torch
from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager


class LEagleTargetInputState:
    def __init__(self, buffers, requests, max_requests, query_len, *, functions=None, h2d=None):
        if functions is None:
            from vllm.v1.worker.gpu.input_batch import (
                combine_sampled_and_draft_tokens,
                expand_idx_mapping,
                prepare_pos_seq_lens,
            )

            functions = (expand_idx_mapping, prepare_pos_seq_lens, combine_sampled_and_draft_tokens)
        if h2d is None:
            from vllm.utils.torch_utils import async_tensor_h2d

            h2d = async_tensor_h2d
        self.buffers, self.requests = buffers, requests
        self.max_requests, self.query_len = max_requests, query_len
        self.functions, self.h2d = functions, h2d
        self.block_tables = None
        query_size = buffers.query_start_loc.numel()
        self.metadata = torch.zeros(
            4 * max_requests + 1 + query_size, dtype=torch.int32, device=buffers.input_ids.device
        )
        self.idx = self.metadata[:max_requests]
        self.cu = self.metadata[max_requests : 2 * max_requests + 1]
        start = 2 * max_requests + 1
        self.query = self.metadata[start : start + query_size]
        self.prefill = self.metadata[start + query_size : start + query_size + max_requests]
        self.block_counts = self.metadata[start + query_size + max_requests :].view(1, max_requests)

    def stage(self, slots, cu_logits, query):
        slots = np.asarray(slots, dtype=np.int32)
        cu_logits = np.asarray(cu_logits, dtype=np.int32)
        query = np.asarray(query, dtype=np.int32)
        count = len(slots)
        if not 0 < count <= self.max_requests:
            raise ValueError("LEAGLE input graph request count exceeds capacity")
        if np.any(slots < 0) or np.any(slots >= self.max_requests):
            raise ValueError("LEAGLE input graph requires real request slots")
        if len(cu_logits) != count + 1 or cu_logits[0] != 0:
            raise ValueError("Invalid LEAGLE logits boundaries")
        if len(query) != self.query.numel() or query[0] != 0 or np.any(np.diff(query[: count + 1]) < 0):
            raise ValueError("Invalid LEAGLE query boundaries")
        lengths = np.diff(cu_logits)
        if np.any(lengths < 1) or np.any(lengths > self.query_len):
            raise ValueError("LEAGLE logits count exceeds captured decode length")
        if not np.array_equal(np.diff(query[: count + 1]), lengths):
            raise ValueError("LEAGLE input graphs require decode-only queries")
        if np.any(query < 0) or query.max() > self.buffers.input_ids.numel():
            raise ValueError("LEAGLE queries exceed input capacity")
        packed = np.zeros(self.metadata.numel(), dtype=np.int32)
        packed[:count] = slots
        packed[self.max_requests : self.max_requests + count + 1] = cu_logits
        start = 2 * self.max_requests + 1
        packed[start : start + len(query)] = query
        packed[start + len(query) : start + len(query) + self.max_requests] = self.requests.prefill_len.np
        if self.block_tables is not None:
            if self.block_tables.num_kv_cache_groups != 1:
                raise ValueError("LEAGLE input graphs require one KV-cache group")
            packed[-self.max_requests :] = self.block_tables.num_blocks.np[0]
        self.h2d(packed, out=self.metadata)
        self.staged = (count, int(cu_logits[-1]))

    def seed(self, count):
        query = np.full(self.query.numel(), count, dtype=np.int32)
        query[: count + 1] = np.arange(count + 1, dtype=np.int32)
        self.stage(np.zeros(count, dtype=np.int32), np.arange(count + 1, dtype=np.int32), query)

    def forward(self, count):
        expand, positions, combine = self.functions
        buffers, requests = self.buffers, self.requests
        buffers.query_start_loc.copy_(self.query)
        slots, cu = self.idx[:count], self.cu[: count + 1]
        expanded, local = expand(slots, count * self.query_len, cu, self.query_len)
        positions(slots, buffers.query_start_loc, requests.num_computed_tokens.gpu, buffers.positions, buffers.seq_lens)
        logits = combine(
            buffers.input_ids,
            slots,
            requests.last_sampled_tokens,
            buffers.query_start_loc,
            buffers.seq_lens,
            self.prefill,
            requests.draft_tokens,
            cu,
            count * self.query_len,
            1,
        )
        if self.block_tables is None:
            return expanded, local, logits
        tables = self.block_tables.gather_block_tables(
            slots, num_reqs_padded=self.max_requests, num_blocks=self.block_counts
        )
        mappings = self.block_tables.compute_slot_mappings(
            slots,
            buffers.query_start_loc,
            buffers.positions,
            num_tokens_padded=buffers.input_ids.numel(),
        )
        return expanded, local, logits, tables, mappings


class LEagleTargetInputGraphRunner(CudaGraphManager):
    """Lifecycle marker lets the framework discard profiling-only captures."""

    def __init__(self, state, backend):
        self.state, self.backend = state, backend
        self.graphs = {}
        self.replays = 0

    @torch.inference_mode()
    def capture(self):
        for count in range(self.state.max_requests, 0, -1):
            if count not in self.graphs:
                self.state.seed(count)
                self.graphs[count] = self.backend.capture(lambda count=count: self.state.forward(count))

    def ready(self):
        return len(self.graphs) == self.state.max_requests

    def replay(self, count, total_logits):
        if not self.ready():
            raise RuntimeError("LEAGLE target input graphs were not captured")
        if (count, total_logits) != self.state.staged:
            raise ValueError("LEAGLE replay shape differs from staged inputs")
        graph, outputs = self.graphs[count]
        graph.replay()
        self.replays += 1
        self.prepared_attn = outputs[3:] if len(outputs) > 3 else None
        return tuple(tensor[:total_logits] for tensor in outputs[:3])
