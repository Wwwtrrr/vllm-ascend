# SPDX-License-Identifier: Apache-2.0
"""CPU test of the reference proposer's NPU sampling binding."""

import ast
import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS

import pytest
import torch


def test_mmuse_kv_option_is_not_forwarded_to_ascend_config():
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
    additional_config = {"mmuse_draft_kv_reuse": True, "enable_fused_mc2": 0}
    forwarded = {key: value for key, value in additional_config.items() if key not in keys}
    assert forwarded == {"enable_fused_mc2": 0}
    assert additional_config["mmuse_draft_kv_reuse"] is True


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
