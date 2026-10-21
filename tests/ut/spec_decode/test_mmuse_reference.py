# SPDX-License-Identifier: Apache-2.0
"""CPU test of the reference proposer's NPU sampling binding."""

import ast
import sys
from pathlib import Path
from types import ModuleType


def test_reference_proposer_uses_ascend_sampler(monkeypatch):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/v2/spec_decode/mmuse/reference.py"
    tree = ast.parse(path.read_text())
    node = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    namespace = {"MMuseReferenceSpeculator": object}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    sampler = ModuleType("vllm_ascend.worker.v2.sample.gumbel")
    sampler.gumbel_sample = lambda *args, **kwargs: (args, kwargs)
    monkeypatch.setitem(sys.modules, sampler.__name__, sampler)
    proposer = namespace[node.name]()
    assert proposer.sample("logits", is_drafting=True) == (("logits",), {"is_drafting": True})
