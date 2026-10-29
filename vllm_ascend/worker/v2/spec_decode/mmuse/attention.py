# SPDX-License-Identifier: Apache-2.0
"""FIA bridges preserving the reference Draft's explicit visibility mask.

This uses the same TorchNPU inference operator family as Ascend attention, but
does not register paged attention or updatable graph tasks. Context ownership
and query/context concatenation remain with the reference decoder.
"""

import torch
from torch import nn

_PAGE_SIZE = 128


def bind_framework_attention(model, config, query_length):
    """Register ordinary cache-owning Attention layers with the worker config."""
    import copy

    from vllm.config import set_current_vllm_config
    from vllm.model_executor.layers.attention import Attention

    from vllm_ascend.attention.attention_v1 import AscendAttentionBackend

    if getattr(config, "kv_transfer_config", None) is not None:
        raise ValueError("MMuse framework FIA does not support KV transfer/offload layouts")
    cache_config = copy.copy(config.cache_config)
    cache_config.cache_dtype = "auto"
    cache_config.sliding_window = None
    cache_config.calculate_kv_scales = False
    # Keep full scheduler ownership even for training's fixed-anchor SWA mask.
    # Applying a per-query SlidingWindowSpec would change its eviction contract.
    with set_current_vllm_config(config):
        for index, layer in enumerate(model.layers):
            attention = layer.self_attn
            owner = Attention(
                attention.heads,
                attention.dim,
                attention.dim**-0.5,
                num_kv_heads=attention.kv_heads,
                cache_config=cache_config,
                quant_config=None,
                attn_backend=AscendAttentionBackend,
                prefix=f"draft_model.layers.{index}.self_attn.attn",
            ).to(device=model.feature_storage.device)
            attention.paged_attention_impl = FrameworkContextAttention(owner, model.feature_storage, query_length)


def fused_context_attention(query, key, value, visible):
    """Map [tokens, heads, dim] GQA and an allowed mask to dense BNSD FIA."""
    import torch_npu  # Lazy: only workers selecting this backend need TorchNPU.

    if query.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("MMuse ascend_fia attention requires FP16 or BF16")
    # SDPA bool True means visible; FIA bool True means excluded. Do not replace
    # this mask with per-query SWA: the training window is anchored to the block.
    excluded = (~visible).contiguous()
    output, _ = torch_npu.npu_fused_infer_attention_score(
        query=query.transpose(0, 1).unsqueeze(0).contiguous(),
        key=key.transpose(0, 1).unsqueeze(0).contiguous(),
        value=value.transpose(0, 1).unsqueeze(0).contiguous(),
        atten_mask=excluded,
        input_layout="BNSD",
        num_heads=query.shape[1],
        num_key_value_heads=key.shape[1],
        scale=query.shape[-1] ** -0.5,
        sparse_mode=0,
        pre_tokens=key.shape[0],
        next_tokens=key.shape[0],
    )
    return output.squeeze(0).transpose(0, 1)


class PagedContextAttention(nn.Module):
    """Native cache-write/paged-read bridge with a reserved per-layer arena.

    One active request reuses this arena, matching the fixed graph's serial
    replay. Pages are not allocated from Target or scheduler-owned block pools.
    Buffers are nonpersistent and allocated before memory profiling/capture.
    """

    def __init__(self, feature_storage, kv_heads, head_dim, query_length):
        super().__init__()
        self.capacity = feature_storage.shape[1]
        self.query_length = query_length
        self.page_size = _PAGE_SIZE
        num_pages = (self.capacity + query_length + self.page_size - 1) // self.page_size
        shape = (num_pages, self.page_size, kv_heads, head_dim)
        self.register_buffer("key_cache", feature_storage.new_zeros(shape), persistent=False)
        self.register_buffer("value_cache", feature_storage.new_zeros(shape), persistent=False)
        table = torch.arange(num_pages, dtype=torch.int32, device=feature_storage.device)[None]
        self.register_buffer("block_table", table, persistent=False)
        logical = torch.arange(self.capacity + query_length, device=feature_storage.device)
        slots = table[0, logical // self.page_size].long() * self.page_size + logical % self.page_size
        self.register_buffer("slots", slots.to(torch.int32), persistent=False)

    def _write(self, key, value, slots):
        # Reuse the framework's platform-dispatched cache writer and its standard
        # [blocks, tokens, KV heads, dim] layout; never copy the cache destination.
        from vllm_ascend.device.device_op import DeviceOperator

        DeviceOperator.reshape_and_cache(
            key,
            value,
            self.key_cache,
            self.value_cache,
            slots,
        )

    def prepare_context(self, key, value, *, start=0):
        length = key.shape[0]
        if not 0 <= start <= length <= self.capacity:
            raise ValueError("MMuse paged context exceeds its reserved capacity")
        if start < length:
            self._write(key[start:], value[start:], self.slots[start:length])

    def forward(self, query, key, value, visible):
        import torch_npu

        length = visible.shape[-1]
        context_length = length - query.shape[0]
        if query.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("MMuse paged FIA requires FP16 or BF16")
        if query.shape[0] != self.query_length or not 0 < context_length <= self.capacity:
            raise ValueError("MMuse paged FIA requires a complete query block and context")
        self._write(key, value, self.slots[context_length:length])
        # PA expects packed H in the dense cache layout; this view aliases the
        # cache passed to the native writer. Actual length excludes unused pages.
        output, _ = torch_npu.npu_fused_infer_attention_score(
            query=query.transpose(0, 1).unsqueeze(0).contiguous(),
            key=self.key_cache.flatten(2),
            value=self.value_cache.flatten(2),
            block_table=self.block_table,
            block_size=self.page_size,
            actual_seq_lengths_kv=[length],
            atten_mask=(~visible).contiguous(),
            input_layout="BNSD",
            num_heads=query.shape[1],
            num_key_value_heads=self.key_cache.shape[2],
            scale=query.shape[-1] ** -0.5,
            sparse_mode=0,
            pre_tokens=length,
            next_tokens=length,
        )
        return output.squeeze(0).transpose(0, 1)


class FrameworkContextAttention(nn.Module):
    """Read/write scheduler-owned Draft pages through a registered Attention.

    Context and query positions are absolute scheduler positions. A padded virtual
    table plus an explicit remapped mask keeps graph shapes fixed without writing
    to unallocated padding pages. Block zero is reserved for capture/padding.
    """

    def __init__(self, cache_owner, feature_storage, query_length):
        super().__init__()
        self.cache_owner = cache_owner
        self.capacity = feature_storage.shape[1]
        self.query_length = query_length
        self.register_buffer("bounds", torch.tensor([0, 1], device=feature_storage.device), persistent=False)
        self.register_buffer(
            "query_slots",
            torch.full((query_length,), -1, dtype=torch.int32, device=feature_storage.device),
            persistent=False,
        )
        self.register_buffer("block_table", None, persistent=False)
        self.register_buffer("offsets", None, persistent=False)
        self.set_page_size(_PAGE_SIZE)
        self._mode = "profile"
        self._source_table = None
        self._span = (0, 1)
        self._profile_context = None
        self.requires_eager = False

    def set_page_size(self, page_size):
        self.page_size = page_size
        max_pages = (self.capacity + page_size - 1 + self.query_length + page_size - 1) // page_size
        device = self.bounds.device
        self.block_table = torch.zeros(1, max_pages, dtype=torch.int32, device=device)
        self.offsets = torch.arange(max_pages * page_size, device=device)

    @property
    def key_cache(self):
        return self.cache_owner.kv_cache[0]

    @property
    def value_cache(self):
        return self.cache_owner.kv_cache[1]

    _write = PagedContextAttention._write

    def _validate_cache(self):
        cache = self.cache_owner.kv_cache
        expected = (self.page_size, self.cache_owner.num_kv_heads, self.cache_owner.head_size)
        if cache.ndim != 5 or cache.shape[0] != 2 or tuple(cache.shape[2:]) != expected:
            raise ValueError("MMuse framework FIA requires standard [2, blocks, tokens, heads, dim] KV layout")
        if cache.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("MMuse framework FIA requires unquantized FP16/BF16 KV")

    def reset_request(self):
        self._mode = "profile"
        self._source_table = None
        self._profile_context = None
        self.requires_eager = False

    def prepare_for_capture(self):
        self._validate_cache()
        self.reset_request()
        self._mode = "capture"
        self.bounds.copy_(self.bounds.new_tensor([0, 1]))
        self.block_table.zero_()
        self.query_slots.fill_(-1)
        # Block zero is the framework's reserved null block, never a live page.
        self.cache_owner.kv_cache[:, 0].zero_()

    def begin_request(self, table, start, end):
        self._validate_cache()
        if not 0 <= start < end or end - start > self.capacity:
            raise ValueError("Invalid MMuse framework context span")
        first_page = start // self.page_size
        last_page = (end + self.query_length + self.page_size - 1) // self.page_size
        if last_page > table.numel():
            self.reset_request()
            self.requires_eager = True
            return
        self._mode = "request"
        self.requires_eager = False
        self._source_table, self._span = table, (start, end)
        self._profile_context = None
        self.bounds.copy_(self.bounds.new_tensor([start, end]))
        self.block_table.zero_()
        self.block_table[0, : last_page - first_page].copy_(table[first_page:last_page])
        positions = torch.arange(end, end + self.query_length, device=table.device)
        slots = table[positions // self.page_size].long() * self.page_size + positions % self.page_size
        self.query_slots.copy_(slots)
        # Masked V rows must also be finite: do not let uninitialized page tails
        # turn a zero attention weight into 0 * NaN. Only this layer's allocated
        # leading/tail fragments are cleared, never another request's pages.
        leading = torch.arange(first_page * self.page_size, start, device=table.device)
        trailing = torch.arange(end + self.query_length, last_page * self.page_size, device=table.device)
        padding = torch.cat((leading, trailing))
        if padding.numel():
            padding_slots = table[padding // self.page_size].long() * self.page_size + padding % self.page_size
            zeros = self.key_cache.new_zeros(padding.numel(), self.cache_owner.num_kv_heads, self.cache_owner.head_size)
            self._write(zeros, zeros, padding_slots.to(torch.int32))

    def prepare_context(self, key, value, *, start=0):
        if self._mode == "profile":
            self._profile_context = (key, value)
            return
        if self._mode == "capture":
            return
        first, end = self._span
        valid_length = end - first
        if key.shape[0] < valid_length or not 0 <= start <= valid_length:
            raise ValueError("MMuse context preparation does not match scheduler span")
        # The dense projection cache can survive scheduler preemption while
        # physical pages are reassigned. Rewrite the valid prefix into the
        # current table; reusing projected K/V does not prove page ownership.
        start = 0
        if start < valid_length:
            positions = torch.arange(first + start, end, device=key.device)
            table = self._source_table
            slots = table[positions // self.page_size].long() * self.page_size + positions % self.page_size
            self._write(key[start:valid_length], value[start:valid_length], slots.to(torch.int32))

    def forward(self, query, key, value, visible):
        if self._mode == "profile":
            if self._profile_context is None:
                raise RuntimeError("MMuse profile context was not prepared")
            context_key, context_value = self._profile_context
            self._profile_context = None
            return fused_context_attention(
                query, torch.cat((context_key, key)), torch.cat((context_value, value)), visible
            )
        import torch_npu

        self._write(key, value, self.query_slots)
        context_size = visible.shape[-1] - query.shape[0]
        virtual_length = (
            (context_size + self.page_size - 1 + query.shape[0] + self.page_size - 1) // self.page_size
        ) * self.page_size
        first, end = self.bounds.unbind()
        absolute = first // self.page_size * self.page_size + self.offsets[:virtual_length]
        in_context = (absolute >= first) & (absolute < end)
        in_query = (absolute >= end) & (absolute < end + query.shape[0])
        source = torch.where(in_context, absolute - first, absolute - end + context_size)
        source = source.clamp(0, visible.shape[-1] - 1).long()
        allowed = visible.index_select(1, source) & (in_context | in_query)[None]
        output, _ = torch_npu.npu_fused_infer_attention_score(
            query=query.transpose(0, 1).unsqueeze(0).contiguous(),
            key=self.key_cache.flatten(2),
            value=self.value_cache.flatten(2),
            block_table=self.block_table,
            block_size=self.page_size,
            actual_seq_lengths_kv=[virtual_length],
            atten_mask=(~allowed).contiguous(),
            input_layout="BNSD",
            num_heads=query.shape[1],
            num_key_value_heads=self.key_cache.shape[2],
            scale=query.shape[-1] ** -0.5,
            sparse_mode=0,
            pre_tokens=virtual_length,
            next_tokens=virtual_length,
        )
        return output.squeeze(0).transpose(0, 1)
