# SPDX-License-Identifier: Apache-2.0
"""Ascend sampling for the independent DeepSeek MMuse block speculator."""

from vllm.v1.worker.gpu.spec_decode.mmuse.deepseek import DeepseekV4MMuseSpeculator


class AscendDeepseekV4MMuseSpeculator(DeepseekV4MMuseSpeculator):
    def _gumbel_sample(self, *args, **kwargs):
        from vllm_ascend.worker.v2.sample.gumbel import gumbel_sample

        return gumbel_sample(*args, **kwargs)
