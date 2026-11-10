# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Private online INT8 full-vocabulary head for BF16 Ascend LEAGLE drafting."""

import torch
from torch import nn


class AscendLEagleInt8Head(nn.Module):
    """Per-output weight scales and per-token activation scales; no disk state.

    Uses npu_dynamic_quant + npu_quant_matmul without retaining the source
    weight. Serving releases the dense draft head after installation.
    Injected ops/formatting allow CPU contract tests only; production
    has no emulation fallback. CANN/NPU support must be validated on hardware.
    """

    @torch.no_grad()
    def __init__(self, weight, *, ops=None, weight_format=None):
        super().__init__()
        if weight.is_meta or weight.ndim != 2 or weight.dtype != torch.bfloat16:
            raise ValueError("LEAGLE W8A8 requires a loaded BF16 LM head")
        if ops is None:
            if weight.device.type != "npu":
                raise ValueError("LEAGLE W8A8 requires Ascend NPU kernels")
            import torch_npu

            from vllm_ascend.utils import maybe_trans_nz

            ops, weight_format = torch_npu, maybe_trans_nz
        if not all(hasattr(ops, name) for name in ("npu_dynamic_quant", "npu_quant_matmul")):
            raise ValueError("Required LEAGLE W8A8 NPU operators are unavailable")
        self.vocab_size, self.hidden_size = weight.shape
        if self.hidden_size % 32 or self.vocab_size % 32:
            raise ValueError("Initial LEAGLE W8A8 path requires hidden/vocab multiples of 32")
        self.dynamic_quant = ops.npu_dynamic_quant
        self.quant_matmul = ops.npu_quant_matmul
        # Allocate directly in [K,N] layout; only chunks are converted to FP32.
        quantized = torch.empty(self.hidden_size, self.vocab_size, dtype=torch.int8, device=weight.device)
        scales = torch.empty(self.vocab_size, dtype=torch.float32, device=weight.device)
        for start in range(0, self.vocab_size, 2048):
            rows = weight[start : start + 2048].detach().float()
            if not torch.isfinite(rows).all():
                raise ValueError("LEAGLE W8A8 requires finite dense weights")
            maximum = rows.abs().amax(-1)
            scale = torch.where(maximum == 0, 1.0, (maximum / 127).clamp_min(torch.finfo(torch.float32).tiny))
            q = (rows / scale[:, None]).round().clamp(-127, 127).to(torch.int8)
            quantized[:, start : start + rows.shape[0]].copy_(q.T)
            scales[start : start + rows.shape[0]].copy_(scale)
        if weight_format is not None:
            quantized = weight_format(quantized)
        self.register_buffer("weight_int8", quantized, persistent=False)
        self.register_buffer("weight_scale", scales, persistent=False)

    def forward(self, hidden):
        if (
            hidden.dtype != torch.bfloat16
            or hidden.shape[-1] != self.hidden_size
            or hidden.device != self.weight_int8.device
        ):
            raise ValueError("LEAGLE W8A8 input must be BF16 with matching hidden size")
        flattened = hidden.reshape(-1, self.hidden_size).contiguous()
        if flattened.shape[0] == 0:
            return hidden.new_empty(*hidden.shape[:-1], self.vocab_size)
        quantized, scale = self.dynamic_quant(flattened, dst_type=torch.int8)
        logits = self.quant_matmul(
            quantized,
            self.weight_int8,
            self.weight_scale,
            pertoken_scale=scale.reshape(-1).float(),
            output_dtype=torch.bfloat16,
        )
        return logits.view(*hidden.shape[:-1], self.vocab_size)
