# SPDX-License-Identifier: Apache-2.0
"""Dense FIA bridge preserving the reference Draft's explicit visibility mask.

This uses the same TorchNPU inference operator family as Ascend attention, but
does not register paged attention or updatable graph tasks. Context ownership
and query/context concatenation remain with the reference decoder.
"""

import torch


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
