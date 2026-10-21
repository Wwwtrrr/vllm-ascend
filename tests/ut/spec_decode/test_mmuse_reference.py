# SPDX-License-Identifier: Apache-2.0
"""CPU test of the reference proposer's NPU sampling binding."""

import ast
import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS

import pytest
import torch


def test_mmuse_trace_option_is_not_forwarded_to_ascend_config():
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/ascend_config.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    factory = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "init_ascend_config"
    )
    excluded = next(
        node.value
        for node in ast.walk(factory)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "_NON_USER_INPUT_KEYS" for target in node.targets)
    )
    keys = eval(compile(ast.Expression(excluded), str(path), "eval"), {"_RUNTIME_CONFIG_STRIP_KEYS": set()})
    additional_config = {"mmuse_trace_steps": 4, "enable_fused_mc2": 0}
    forwarded = {key: value for key, value in additional_config.items() if key not in keys}
    assert forwarded == {"enable_fused_mc2": 0}
    assert additional_config["mmuse_trace_steps"] == 4


def test_reference_proposer_uses_ascend_sampler(monkeypatch):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/v2/spec_decode/mmuse/reference.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    namespace = {"MMuseReferenceSpeculator": object}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    sampler = ModuleType("vllm_ascend.worker.v2.sample.gumbel")
    sampler.gumbel_sample = lambda *args, **kwargs: (args, kwargs)
    monkeypatch.setitem(sys.modules, sampler.__name__, sampler)
    proposer = namespace[node.name]()
    assert proposer.sample("logits", is_drafting=True) == (("logits",), {"is_drafting": True})


def test_quarot_resolves_bundled_matrix_and_rejects_missing_file(monkeypatch, tmp_path):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/v2/spec_decode/mmuse/reference.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    namespace = {"MMuseReferenceSpeculator": object}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)
    proposer = namespace[cls.name]()
    root = tmp_path / "model"
    bundled = root / "optional" / "quarot.safetensors"
    bundled.parent.mkdir(parents=True)
    bundled.touch()
    utils = ModuleType("vllm_ascend.utils")
    utils.get_rotation_path = lambda _: root / ".." / "optional" / bundled.name
    seen = []
    rotation = torch.eye(4)
    utils.get_rotation_matrix = lambda p: (seen.append(p), rotation)[1]
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    proposer.vllm_config = NS(model_config=NS(model=str(root)))
    assert proposer.get_target_rotation() is rotation
    assert seen == [bundled]
    bundled.unlink()
    with pytest.raises(ValueError, match="matrix is missing"):
        proposer.get_target_rotation()
    utils.get_rotation_path = lambda _: None
    assert proposer.get_target_rotation() is None


def test_lmhead_tp_trace_receives_trimmed_logits_before_verification():
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/v2/model_runner.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "sample")
    namespace = dict(lmhead_tp_enable=lambda: True, lmhead_tp_pad_rows=lambda *args: torch.tensor([0, 1, 0, 0]))
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    events = []
    batch = NS(logits_indices=torch.tensor([0, 1]), num_draft_tokens=1)
    output = NS(num_sampled=torch.tensor([1]), num_rejected=torch.tensor([1]))

    def verify(logits, actual_batch, draft_logits):
        events.append(("verify", tuple(logits.shape)))
        assert actual_batch is batch
        return output

    runner = NS(
        _lmhead_tp_max_num_logits=lambda: 4,
        model=NS(compute_logits=lambda hidden: torch.zeros(hidden.shape[0], 8)),
        speculator=NS(draft_logits=None, trace_target=lambda logits, _: events.append(("trace", tuple(logits.shape)))),
        rejection_sampler=verify,
    )
    result = namespace["sample"](runner, torch.zeros(4, 3), batch, None)
    assert events == [("trace", (2, 8)), ("verify", (2, 8))]
    assert result[0] is output
