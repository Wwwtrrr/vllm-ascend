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
    additional_config = {"mmuse_draft_kv_reuse": True, "mmuse_attention_backend": "ascend_fia", "enable_fused_mc2": 0}
    forwarded = {key: value for key, value in additional_config.items() if key not in keys}
    assert forwarded == {"enable_fused_mc2": 0}
    assert additional_config["mmuse_draft_kv_reuse"] is True


@pytest.mark.parametrize("backend", ["reference", "ascend_fia", "ascend_paged_fia", "ascend_framework_fia", "unknown"])
def test_attention_binding_is_explicit_and_preserves_reference(monkeypatch, backend):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/v2/spec_decode/mmuse/reference.py"
    cls = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef))
    namespace = {"MMuseReferenceSpeculator": object}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)
    module = ModuleType("vllm_ascend.worker.v2.spec_decode.mmuse.attention")
    module.fused_context_attention = object()
    page_backend = object()
    module.PagedContextAttention = lambda *args: page_backend
    framework_calls = []
    module.bind_framework_attention = lambda *args: framework_calls.append(args)
    primitives = ModuleType("vllm.model_executor.models.mmuse.framework_decoder")
    primitive_calls = []
    primitives.bind_framework_decoder = lambda *args: primitive_calls.append(args)
    monkeypatch.setitem(sys.modules, primitives.__name__, primitives)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    proposer = namespace[cls.name]()
    proposer.num_query_per_req = 3
    proposer.vllm_config = NS(additional_config={"mmuse_attention_backend": backend})
    attention = NS(attention_impl=None, paged_attention_impl=None, kv_heads=2, dim=64)
    model = NS(layers=[NS(self_attn=attention)], feature_storage=torch.zeros(1, 16, 128))
    if backend == "unknown":
        with pytest.raises(ValueError, match="backend"):
            proposer.configure_attention(model)
    else:
        proposer.configure_attention(model)
        expected = module.fused_context_attention if backend == "ascend_fia" else None
        assert attention.attention_impl is expected
        assert attention.paged_attention_impl is (page_backend if backend == "ascend_paged_fia" else None)
        assert len(framework_calls) == (1 if backend == "ascend_framework_fia" else 0)
        assert len(primitive_calls) == len(framework_calls)


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
