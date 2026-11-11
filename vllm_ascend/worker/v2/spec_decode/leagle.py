# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Ascend LEAGLE uses the same checkpoint and recurrence as CUDA."""

import torch
from vllm.v1.worker.gpu.spec_decode.leagle.graph import NativeLEagleGraphBackend
from vllm.v1.worker.gpu.spec_decode.leagle.speculator import LEagleSpeculator


class AscendLEagleSpeculator(LEagleSpeculator):
    """PyTorch SDPA on NPU, with native ACL FULL graphs for the drafter."""

    def configure_lm_head_quantization(self):
        mode = getattr(self.speculative_config, "leagle_lm_head_quantization", None)
        if mode is None:
            return
        if mode != "w8a8":
            raise ValueError("Ascend LEAGLE only supports w8a8 full-head quantization")
        from vllm_ascend.worker.v2.spec_decode.leagle_quant import AscendLEagleInt8Head

        release = getattr(self.speculative_config, "leagle_release_quantized_source", False)
        if self.model.svd_candidate_head is not None:
            self.model.set_quantized_svd_head(
                AscendLEagleInt8Head(self.model.svd_candidate_head.up_weight),
                release_source=release,
            )
            return
        if self.model.candidate_head_enabled:
            raise ValueError("LEAGLE W8A8 requires SVD for candidate scoring")
        self.model.set_quantized_lm_head(
            AscendLEagleInt8Head(self.model.lm_head.weight), release_source=release
        )

    def create_graph_backend(self):
        return NativeLEagleGraphBackend(torch.npu, self.device, torch.npu.NPUGraph)

    def create_rmsnorm_impl(self):
        def rms_norm(hidden, weight, epsilon):
            flattened = hidden.reshape(-1, hidden.shape[-1]).contiguous()
            return torch.ops.npu.npu_rms_norm(flattened, weight, epsilon)[0].view_as(hidden)

        return rms_norm
