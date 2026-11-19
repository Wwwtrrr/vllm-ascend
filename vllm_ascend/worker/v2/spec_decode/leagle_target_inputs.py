# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Prepare LEAGLE target KV inputs inside the target's FULL graph."""

import numpy as np
import torch
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

from vllm_ascend.ops.rotary_embedding import update_cos_sin
from vllm_ascend.worker.v2.block_table import AscendBlockTables
from vllm_ascend.worker.v2.input_batch import AscendInputBuffers


@triton.jit
def _gather_dense_block_rows(
    idx_mapping,
    seq_lens,
    source,
    output,
    NUM_BLOCKS: tl.constexpr,
    SOURCE_STRIDE: tl.constexpr,
    OUTPUT_STRIDE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    column = tl.program_id(1) * TILE_SIZE + tl.arange(0, TILE_SIZE)
    request_index = tl.load(idx_mapping + row)
    is_active = request_index >= 0
    seq_len = tl.load(seq_lens + row, mask=is_active, other=0)
    # The dense target reads only pages covered by the exact device length.
    # Do not capture the address of the rotating host-visible block counts.
    num_live_blocks = tl.cdiv(seq_len, PAGE_SIZE)
    block_ids = tl.load(
        source + tl.maximum(request_index, 0) * SOURCE_STRIDE + column,
        mask=is_active & (column < NUM_BLOCKS) & (column < num_live_blocks),
        other=0,
    )
    tl.store(output + row * OUTPUT_STRIDE + column, block_ids, mask=column < NUM_BLOCKS)


class LEagleTargetInputGraph:
    """Capture a prefix that writes into the target's existing KV buffers."""

    def __init__(self, block_tables: AscendBlockTables, input_buffers: AscendInputBuffers) -> None:
        self.block_tables = block_tables
        self.input_buffers = input_buffers
        self.idx_mapping = torch.full_like(input_buffers.seq_lens, -1)
        self._padded_idx_mapping: np.ndarray | None = None

    def stage(self, idx_mapping_np: np.ndarray, num_graph_reqs: int) -> torch.Tensor:
        num_reqs = len(idx_mapping_np)
        if not num_reqs <= num_graph_reqs <= self.idx_mapping.numel():
            raise ValueError("LEAGLE target input requests exceed the graph capacity.")

        indices = idx_mapping_np
        if num_reqs < num_graph_reqs:
            if self._padded_idx_mapping is None:
                self._padded_idx_mapping = np.empty(self.idx_mapping.numel(), dtype=np.int32)
            indices = self._padded_idx_mapping[:num_graph_reqs]
            indices[:num_reqs] = idx_mapping_np
            indices[num_reqs:] = -1

        async_tensor_h2d(indices, out=self.idx_mapping[:num_graph_reqs])
        return self.idx_mapping[:num_reqs]

    def clear_dummy(self) -> None:
        self.idx_mapping.fill_(-1)

    def capture(self, desc: BatchExecutionDescriptor) -> None:
        num_reqs = desc.num_reqs
        assert num_reqs is not None and 0 < num_reqs <= self.idx_mapping.numel()
        move_tile = 256
        for table, output, page_size in zip(
            self.block_tables.block_tables,
            self.block_tables.input_block_tables,
            self.block_tables.kernel_block_sizes,
        ):
            source = table.gpu
            num_blocks = output.shape[1]
            _gather_dense_block_rows[(num_reqs, triton.cdiv(num_blocks, move_tile))](
                self.idx_mapping,
                self.input_buffers.seq_lens,
                source,
                output,
                num_blocks,
                source.stride(0),
                output.stride(0),
                page_size,
                move_tile,
            )

        positions = self.input_buffers.positions[: desc.num_tokens]
        self.block_tables.compute_slot_mappings(
            self.idx_mapping[:num_reqs],
            self.input_buffers.query_start_loc,
            positions,
            desc.num_tokens,
        )
        update_cos_sin(positions)
