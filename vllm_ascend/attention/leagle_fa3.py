# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Ascend FA3 with device lengths for the Qwen3 target of LEAGLE.

Adapted from vllm-ascend#21444 (SkychenLee).
"""

from dataclasses import dataclass, field

import torch
from vllm.config import VllmConfig
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
)
from vllm.v1.kv_cache_interface import AttentionSpec

from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.device.device_op import DeviceOperator

FA3_PAGE_SIZE = 128
_HeadLayout = tuple[int, int, int]


class AscendLEagleFA3Backend(AttentionBackend):
    accept_output_buffer = True
    forward_includes_kv_cache_update = True
    supported_dtypes = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes = ["auto", "float16", "bfloat16"]

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return 0 < head_size <= 256

    @staticmethod
    def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str=""):
        return 2, num_blocks, block_size, num_kv_heads, head_size

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int]:
        return [FA3_PAGE_SIZE]

    @staticmethod
    def get_impl_cls():
        return AscendLEagleFA3Impl

    @staticmethod
    def get_builder_cls():
        return AscendLEagleFA3MetadataBuilder


@dataclass
class _GraphBuffers:
    query_offsets: torch.Tensor
    kv_lengths: torch.Tensor
    page_table: torch.Tensor
    tiling: dict[_HeadLayout, torch.Tensor] = field(default_factory=dict)


@dataclass
class AscendLEagleFA3Metadata(AttentionMetadata):
    query_offsets: torch.Tensor
    kv_lengths: torch.Tensor
    page_table: torch.Tensor
    slot_mapping: torch.Tensor
    max_query_len: int
    window: tuple[int, int]
    tiling: dict[_HeadLayout, torch.Tensor]


class AscendLEagleFA3MetadataBuilder(AttentionMetadataBuilder[AscendLEagleFA3Metadata]):
    # The FULL target graph accepts mixed queries and different actual request
    # counts within a fixed token capacity. The builder refreshes its tiling.
    _cudagraph_support = AttentionCGSupport.ALWAYS

    def __init__(
        self, kv_cache_spec: AttentionSpec, layer_names: list[str], vllm_config: VllmConfig, device: torch.device
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        layers = vllm_config.compilation_config.static_forward_context
        self.head_layouts = tuple(dict.fromkeys(layers[name].impl.head_layout for name in layer_names))
        # Each builder owns its buffers. Layers with equal head geometry in
        # this cache group share tiling. Other groups and graph sizes cannot
        # overwrite metadata captured at a different address.
        self.graph_buffers: dict[int, _GraphBuffers] = {}

    def _stage_graph_inputs(
        self, query_offsets, kv_lengths, page_table, num_input_tokens, num_actual_tokens
    ) -> _GraphBuffers:
        req_capacity = min(self.max_num_reqs, num_input_tokens)
        num_reqs, num_pages = page_table.shape
        buffers = self.graph_buffers.get(num_input_tokens)
        if buffers is None:
            buffers = _GraphBuffers(
                query_offsets.new_empty(req_capacity + 1),
                kv_lengths.new_empty(req_capacity),
                page_table.new_zeros((req_capacity, num_pages + 1)),
            )
            self.graph_buffers[num_input_tokens] = buffers
        if buffers.page_table.shape[1] != num_pages + 1:
            raise RuntimeError("LEAGLE FA3 page table width changed for an existing FULL graph")

        buffers.query_offsets[: num_reqs + 1].copy_(query_offsets)
        buffers.kv_lengths[:num_reqs].copy_(kv_lengths)
        buffers.page_table[:num_reqs, :-1].copy_(page_table)
        if num_reqs < req_capacity:
            buffers.query_offsets[num_reqs + 1 :].fill_(num_actual_tokens)
            buffers.kv_lengths[num_reqs:].zero_()
            buffers.page_table[num_reqs:].zero_()
        return buffers

    def build_for_cudagraph_capture(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        *,
        num_actual_reqs: int | None = None,
        full_graph_mode: bool = True,
    ) -> AscendLEagleFA3Metadata:
        return self.build(0, common_attn_metadata, num_actual_reqs=num_actual_reqs, full_graph_mode=full_graph_mode)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        fast_build: bool = False,
        *,
        num_actual_reqs: int | None = None,
        full_graph_mode: bool = False,
    ) -> AscendLEagleFA3Metadata:
        # The extension is optional and loaded only for a selected backend.
        from flash_attn_npu_3 import get_scheduler_metadata

        common = common_attn_metadata
        num_actual_tokens = common.num_actual_tokens
        num_input_tokens = common.num_input_tokens or num_actual_tokens
        num_reqs = common.num_reqs if num_actual_reqs is None else num_actual_reqs
        # BlockTables uses the same persistent slot buffer during capture and
        # replay. Its padding slots are -1, so fixed-shape writes skip them.
        num_slots = num_input_tokens if full_graph_mode else num_actual_tokens
        if (
            common.causal is not True
            or not 0 < num_reqs <= min(common.num_reqs, self.max_num_reqs, num_actual_tokens)
            or num_actual_tokens > num_input_tokens
            or common.query_start_loc.numel() < num_reqs + 1
            or common.seq_lens.numel() < num_reqs
            or common.block_table_tensor.shape[0] < num_reqs
            or common.slot_mapping.numel() < num_slots
        ):
            raise ValueError("Invalid dense decoder metadata for LEAGLE FA3")

        # Exclude the synthetic rows that target FULL replay can add for FIA.
        # These slices use host-known request/token counts, not host KV lengths.
        query_offsets = common.query_start_loc[: num_reqs + 1]
        kv_lengths = common.seq_lens[:num_reqs]
        page_table = common.block_table_tensor[:num_reqs]
        slots = common.slot_mapping[:num_slots]
        page_size = self.kernel_block_size or self.kv_cache_spec.block_size
        if page_size != FA3_PAGE_SIZE:
            raise ValueError("LEAGLE FA3 requires 128-token KV pages")

        num_pages = page_table.shape[1]
        # The left window covers all possible real keys. The spare page-table
        # column keeps FA3 0.5.0 on its device-length window branch instead of
        # selecting the global TND branch, which reads lengths on the host.
        window = (num_pages * page_size - 1, 0)
        if full_graph_mode:
            buffers = self._stage_graph_inputs(
                query_offsets, kv_lengths, page_table, num_input_tokens, num_actual_tokens
            )
            max_query_len = num_input_tokens
        else:
            padded_table = page_table.new_zeros((num_reqs, num_pages + 1))
            padded_table[:, :-1].copy_(page_table)
            buffers = _GraphBuffers(query_offsets, kv_lengths, padded_table)
            max_query_len = common.max_query_len

        for layout in self.head_layouts:
            num_heads, num_kv_heads, head_size = layout
            # Keep AICPU tiling outside capture, where its auxiliary stream
            # events can run normally. It reads only the active device prefix.
            tiling = get_scheduler_metadata(
                batch_size=num_reqs,
                max_seqlen_q=max_query_len,
                max_seqlen_k=(num_pages + 1) * page_size,
                num_heads_q=num_heads,
                num_heads_kv=num_kv_heads,
                headdim=head_size,
                cache_seqlens=kv_lengths,
                qkv_dtype=self.kv_cache_spec.dtype,
                cu_seqlens_q=query_offsets,
                page_size=page_size,
                causal=True,
                window_size=window,
                has_softcap=False,
                num_splits=0,
            )
            captured_tiling = buffers.tiling.get(layout)
            if captured_tiling is None:
                buffers.tiling[layout] = tiling
            else:
                if (captured_tiling.shape, captured_tiling.dtype) != (tiling.shape, tiling.dtype):
                    raise RuntimeError("LEAGLE FA3 tiling layout changed for an existing FULL graph")
                captured_tiling.copy_(tiling)

        return AscendLEagleFA3Metadata(
            buffers.query_offsets,
            buffers.kv_lengths,
            buffers.page_table,
            slots,
            max_query_len,
            window,
            buffers.tiling,
        )


class AscendLEagleFA3Impl(AttentionImpl[AscendLEagleFA3Metadata]):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
        **kwargs,
    ):
        if (
            attn_type != AttentionType.DECODER
            or alibi_slopes is not None
            or sliding_window is not None
            or logits_soft_cap not in (None, 0.0)
            or kv_sharing_target_layer_name is not None
            or sinks is not None
            or kv_cache_dtype not in AscendLEagleFA3Backend.supported_kv_cache_dtypes
        ):
            raise ValueError("LEAGLE FA3 requires unquantized dense decoder attention")
        self.num_heads = num_heads
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.head_size = head_size
        self.head_layout = (num_heads, self.num_kv_heads, head_size)
        self.scale = float(scale)

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor | None,
        value: torch.Tensor | None,
        kv_cache: torch.Tensor,
        attn_metadata: AscendLEagleFA3Metadata | None,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None
        assert output_scale is None and output_block_scale is None
        if attn_metadata is None:
            return output.zero_()

        from flash_attn_npu_3 import flash_attn_with_kvcache

        assert layer._k_scale_float == 1.0 and layer._v_scale_float == 1.0
        if key is not None and value is not None:
            slot_mapping = attn_metadata.slot_mapping
            DeviceOperator.reshape_and_cache(
                key=key[: slot_mapping.numel()],
                value=value[: slot_mapping.numel()],
                key_cache=kv_cache[0],
                value_cache=kv_cache[1],
                slot_mapping=slot_mapping,
            )
        result = flash_attn_with_kvcache(
            query,
            kv_cache[0],
            kv_cache[1],
            cache_seqlens=attn_metadata.kv_lengths,
            page_table=attn_metadata.page_table,
            cu_seqlens_q=attn_metadata.query_offsets,
            max_seqlen_q=attn_metadata.max_query_len,
            softmax_scale=self.scale,
            causal=True,
            window_size=attn_metadata.window,
            softcap=0.0,
            scheduler_metadata=attn_metadata.tiling[self.head_layout],
            num_splits=0,
        )
        output.copy_(result)
        return output
