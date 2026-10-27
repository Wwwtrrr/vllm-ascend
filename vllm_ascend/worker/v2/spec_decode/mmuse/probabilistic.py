# SPDX-License-Identifier: Apache-2.0
"""Probabilistic MMuse using our independent ACL graph lifecycle."""

from vllm.v1.worker.gpu.spec_decode.mmuse.probabilistic import MMuseProbabilisticSampling

from vllm_ascend.worker.v2.spec_decode.mmuse.greedy import AscendMMuseGreedySpeculator


class AscendMMuseProbabilisticSpeculator(MMuseProbabilisticSampling, AscendMMuseGreedySpeculator):
    def __init__(self, vllm_config, device):
        if vllm_config.model_config.use_fp64_gumbel:
            raise ValueError("MMuse NPU probabilistic drafting does not support FP64 Gumbel")
        super().__init__(vllm_config, device)

    def _gumbel_sample(self, *args, **kwargs):
        from vllm_ascend.worker.v2.sample.gumbel import gumbel_sample

        return gumbel_sample(*args, **kwargs)
