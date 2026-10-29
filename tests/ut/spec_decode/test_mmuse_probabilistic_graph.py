# SPDX-License-Identifier: Apache-2.0
"""Ascend probability graph adapter bindings without an NPU dependency."""

import ast
import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS

import pytest


def proposer_class():
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/v2/spec_decode/mmuse/probabilistic.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))

    class Sampling:
        def __init__(self, config, device):
            self.device = device

    class GraphDriver:
        pass

    scope = {"MMuseProbabilisticSampling": Sampling, "AscendMMuseGreedySpeculator": GraphDriver}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), scope)
    return scope[cls.name]


def test_probability_graph_uses_npu_sampling_and_rejects_fp64(monkeypatch):
    cls = proposer_class()
    config = NS(model_config=NS(use_fp64_gumbel=True))
    with pytest.raises(ValueError, match="FP64"):
        cls(config, "npu")
    config.model_config.use_fp64_gumbel = False
    obj = cls(config, "npu")
    assert obj.device == "npu"
    sampler = ModuleType("vllm_ascend.worker.v2.sample.gumbel")
    sampler.gumbel_sample = lambda *a, **k: (a, k)
    monkeypatch.setitem(sys.modules, sampler.__name__, sampler)
    assert obj._gumbel_sample("scores", is_drafting=True) == (("scores",), {"is_drafting": True})
