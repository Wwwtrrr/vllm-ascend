# SPDX-License-Identifier: Apache-2.0
"""DeepSeek native graph backend and teacher-space binding contracts."""

import ast
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


def test_native_graph_adapter_keeps_quarot_and_npu_sampling_bindings():
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/v2/spec_decode/mmuse/deepseek_full.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))

    class Base:
        def __init__(self, config, device):
            self.device = device

    class Reference:
        def get_target_rotation(self):
            return "declared-quarot"

        def sample(self, *args, **kwargs):
            return args, kwargs

    backend = object()
    scope = {"torch": NS(npu=backend), "DeepseekMMuseFullSpeculator": Base, "AscendMMuseReferenceSpeculator": Reference}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), scope)
    config = NS(model_config=NS(use_fp64_gumbel=False))
    obj = scope[cls.name](config, "npu")
    assert obj.graph_backend() is backend
    assert obj.get_target_rotation() == "declared-quarot"
    assert obj.sample("logits", is_drafting=True) == (("logits",), {"is_drafting": True})
    config.model_config.use_fp64_gumbel = True
    with pytest.raises(ValueError, match="FP64"):
        scope[cls.name](config, "npu")
