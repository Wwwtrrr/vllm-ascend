# SPDX-License-Identifier: Apache-2.0
"""Native NPU query graphs with the existing DeepSeek teacher IO/QuaRot contract."""

import torch
from vllm.v1.worker.gpu.spec_decode.mmuse.deepseek_full import DeepseekMMuseFullSpeculator

from vllm_ascend.worker.v2.spec_decode.mmuse.reference import AscendMMuseReferenceSpeculator


class AscendDeepseekMMuseFullSpeculator(DeepseekMMuseFullSpeculator):
    def __init__(self, config, device):
        if config.model_config.use_fp64_gumbel:
            raise ValueError("DeepSeek MMuse NPU graphs do not support FP64 Gumbel")
        super().__init__(config, device)

    get_target_rotation = AscendMMuseReferenceSpeculator.get_target_rotation
    sample = AscendMMuseReferenceSpeculator.sample
    configure_attention = AscendMMuseReferenceSpeculator.configure_attention

    def graph_backend(self):
        return torch.npu
