# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Ascend eager LEAGLE uses the same checkpoint and recurrence as CUDA."""

from vllm.v1.worker.gpu.spec_decode.leagle.speculator import LEagleSpeculator


class AscendLEagleSpeculator(LEagleSpeculator):
    """PyTorch SDPA on NPU; no ACL graph or paged draft KV is used."""
