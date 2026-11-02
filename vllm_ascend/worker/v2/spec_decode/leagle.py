# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Ascend LEAGLE uses the same checkpoint and recurrence as CUDA."""

import torch
from vllm.v1.worker.gpu.spec_decode.leagle.graph import NativeLEagleGraphBackend
from vllm.v1.worker.gpu.spec_decode.leagle.speculator import LEagleSpeculator


class AscendLEagleSpeculator(LEagleSpeculator):
    """PyTorch SDPA on NPU, with native ACL FULL graphs for the drafter."""

    def create_graph_backend(self):
        return NativeLEagleGraphBackend(torch.npu, self.device, torch.npu.NPUGraph)
