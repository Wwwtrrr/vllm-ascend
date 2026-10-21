# SPDX-License-Identifier: Apache-2.0
"""NPU sampler binding for the stateless MMuse reference drafter."""

from vllm.v1.worker.gpu.spec_decode.mmuse.reference import MMuseReferenceSpeculator


class AscendMMuseReferenceSpeculator(MMuseReferenceSpeculator):
    def sample(self, *args, **kwargs):
        from vllm_ascend.worker.v2.sample.gumbel import gumbel_sample

        return gumbel_sample(*args, **kwargs)
