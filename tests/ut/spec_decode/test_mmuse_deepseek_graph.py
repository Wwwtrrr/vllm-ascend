# SPDX-License-Identifier: Apache-2.0
"""DeepSeek native graph backend and teacher-space binding contracts."""

import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS

import pytest
import torch


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

        def configure_attention(self, model):
            return model

    backend = object()
    scope = {"torch": NS(npu=backend), "DeepseekMMuseFullSpeculator": Base, "AscendMMuseReferenceSpeculator": Reference}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), scope)
    config = NS(model_config=NS(use_fp64_gumbel=False))
    obj = scope[cls.name](config, "npu")
    assert obj.graph_backend() is backend
    assert obj.get_target_rotation() == "declared-quarot"
    assert obj.sample("logits", is_drafting=True) == (("logits",), {"is_drafting": True})
    assert obj.configure_attention("draft") == "draft"
    config.model_config.use_fp64_gumbel = True
    with pytest.raises(ValueError, match="FP64"):
        scope[cls.name](config, "npu")


def attention_bridge():
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/v2/spec_decode/mmuse/attention.py"
    spec = importlib.util.spec_from_file_location("mmuse_fia_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.fused_context_attention


@pytest.mark.parametrize("query_heads,kv_heads", [(4, 2), (2, 2)])
@pytest.mark.parametrize("device", ["cpu", "npu"])
@torch.inference_mode()
def test_fia_preserves_gqa_and_explicit_anchor_mask(monkeypatch, query_heads, kv_heads, device):
    if device == "npu":
        pytest.importorskip("torch_npu")
        if not torch.npu.is_available():
            pytest.skip("NPU required")
    else:
        # CPU oracle validates the bridge's layout, GQA and mask polarity.
        # It does not stand in for native FIA numerical validation.
        fake = ModuleType("torch_npu")

        def fia(**kwargs):
            assert kwargs["input_layout"] == "BNSD"
            assert kwargs["sparse_mode"] == 0
            assert kwargs["num_heads"] == query_heads
            assert kwargs["num_key_value_heads"] == kv_heads
            assert kwargs["scale"] == 64**-0.5
            assert kwargs["pre_tokens"] == kwargs["next_tokens"] == 12
            groups = query_heads // kv_heads
            return torch.nn.functional.scaled_dot_product_attention(
                kwargs["query"],
                kwargs["key"].repeat_interleave(groups, 1),
                kwargs["value"].repeat_interleave(groups, 1),
                attn_mask=~kwargs["atten_mask"],
            ), None

        fake.npu_fused_infer_attention_score = fia
        monkeypatch.setitem(sys.modules, "torch_npu", fake)
    torch.manual_seed(81)
    q = torch.randn(7, query_heads, 64, dtype=torch.bfloat16, device=device)
    k = torch.randn(12, kv_heads, 64, dtype=q.dtype, device=device)
    v = torch.randn_like(k)
    visible = torch.ones(7, 12, dtype=torch.bool, device=device)
    visible[:, :2] = False  # A fixed anchor window, not query-relative SWA.
    visible[:, 4] = False  # Padded context.
    visible[:, 5:] = torch.ones(7, 7, dtype=torch.bool, device=device).tril()
    groups = query_heads // kv_heads
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(0, 1)[None],
        k.transpose(0, 1).repeat_interleave(groups, 0)[None],
        v.transpose(0, 1).repeat_interleave(groups, 0)[None],
        attn_mask=visible,
    )[0].transpose(0, 1)
    actual = attention_bridge()(q, k, v, visible)
    torch.testing.assert_close(actual, expected, atol=0.03, rtol=0.03)
    if device == "npu":
        graph = torch.npu.NPUGraph()
        for _ in range(3):
            attention_bridge()(q, k, v, visible)
        torch.npu.synchronize()
        bridge = attention_bridge()
        with torch.npu.graph(graph):
            output = bridge(q, k, v, visible)
        q.add_(0.1)
        visible[:, 3] = False
        expected = bridge(q, k, v, visible)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(output, expected, atol=0.03, rtol=0.03)
