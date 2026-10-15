# SPDX-License-Identifier: Apache-2.0
"""Ascend DFlash execution with shared reference MMuse heads and rollout."""

from vllm.model_executor.models.mmuse_runtime import (
    MMuseGreedyRolloutMixin,
    validate_mmuse_config,
)

from vllm_ascend.worker.v2.spec_decode.dflash.speculator import AscendDFlashSpeculator


class AscendMMuseSpeculator(MMuseGreedyRolloutMixin, AscendDFlashSpeculator):
    _speculator_name = "MMuse"

    def __init__(self, vllm_config, device):
        validate_mmuse_config(vllm_config)
        super().__init__(vllm_config, device)
        self.sample_from_anchor = True
        self.num_query_per_req = self.num_speculative_steps

    def load_draft_model(self, target_model, target_attn_layer_names):
        model = super().load_draft_model(target_model, target_attn_layer_names)
        model.bind_target_weights(target_model)
        return model
