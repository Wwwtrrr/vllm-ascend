# SPDX-License-Identifier: Apache-2.0
"""NPU sampling and QuaRot binding for the MMuse reference drafter."""

from vllm.v1.worker.gpu.spec_decode.mmuse.reference import MMuseReferenceSpeculator


class AscendMMuseReferenceSpeculator(MMuseReferenceSpeculator):
    def configure_attention(self, model):
        backend = (getattr(self.vllm_config, "additional_config", None) or {}).get(
            "mmuse_attention_backend", "reference"
        )
        if backend == "reference":
            return
        if backend != "ascend_fia":
            raise ValueError("MMuse attention backend must be reference or ascend_fia")
        from vllm_ascend.worker.v2.spec_decode.mmuse.attention import fused_context_attention

        for layer in model.layers:
            layer.self_attn.attention_impl = fused_context_attention

    def get_target_rotation(self):
        from pathlib import Path

        from vllm_ascend.utils import get_rotation_matrix, get_rotation_path

        declared = get_rotation_path(self.vllm_config)
        if declared is None:
            return None
        root = Path(self.vllm_config.model_config.model)
        # Some exports use ../optional while shipping optional/ inside the
        # model directory. Prefer the declared layout, then the bundled file.
        path = declared if declared.is_file() else root / "optional" / declared.name
        if not path.is_file():
            raise ValueError(
                f"MMuse Target declares QuaRot but its matrix is missing: {declared} (also checked {path})"
            )
        return get_rotation_matrix(path)

    def sample(self, *args, **kwargs):
        from vllm_ascend.worker.v2.sample.gumbel import gumbel_sample

        return gumbel_sample(*args, **kwargs)
