# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Ascend LEAGLE uses the same checkpoint and recurrence as CUDA."""

import torch
from vllm.v1.worker.gpu.spec_decode.leagle.graph import NativeLEagleGraphBackend
from vllm.v1.worker.gpu.spec_decode.leagle.speculator import LEagleSpeculator


class AscendLEagleSpeculator(LEagleSpeculator):
    """PyTorch SDPA on NPU, with native ACL FULL graphs for the drafter."""

    def bind_target_input_graphs(self, buffers, requests, max_requests, query_len):
        if not getattr(self.speculative_config, "leagle_target_input_graphs", True):
            return
        from vllm_ascend.worker.v2.spec_decode.leagle_inputs import LEagleTargetInputState

        self.target_input_graph_state = LEagleTargetInputState(buffers, requests, max_requests, query_len)
        self.target_input_graph_runner = None

    def init_cudagraph_manager(self, cudagraph_mode):
        super().init_cudagraph_manager(cudagraph_mode)
        state = getattr(self, "target_input_graph_state", None)
        if (
            state is not None
            and cudagraph_mode.has_full_cudagraphs()
            and state.block_tables is not None
            and state.block_tables.num_kv_cache_groups == 1
        ):
            from vllm_ascend.worker.v2.spec_decode.leagle_inputs import LEagleTargetInputGraphRunner

            self.target_input_graph_runner = LEagleTargetInputGraphRunner(state, self.create_graph_backend())

    def set_attn(self, model_state, kv_cache_config, block_tables, target_input_buffers, target_attn_groups):
        super().set_attn(model_state, kv_cache_config, block_tables, target_input_buffers, target_attn_groups)
        state = getattr(self, "target_input_graph_state", None)
        if state is not None:
            state.block_tables = block_tables

    def capture(self):
        super().capture()
        inputs = getattr(self, "target_input_graph_runner", None)
        if inputs is not None:
            inputs.capture()

    def create_graph_backend(self):
        return NativeLEagleGraphBackend(torch.npu, self.device, torch.npu.NPUGraph)

    def create_rmsnorm_impl(self):
        def rms_norm(hidden, weight, epsilon):
            flattened = hidden.reshape(-1, hidden.shape[-1]).contiguous()
            return torch.ops.npu.npu_rms_norm(flattened, weight, epsilon)[0].view_as(hidden)

        return rms_norm
