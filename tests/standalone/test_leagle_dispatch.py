# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check Ascend LEAGLE routing without requiring an NPU runtime."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


def test_leagle_dispatch_keeps_its_own_speculator(monkeypatch):
    config_module = ModuleType("vllm.config")
    config_module.VllmConfig = object
    config_module.CompilationMode = SimpleNamespace(NONE=0)
    monkeypatch.setitem(sys.modules, "vllm.config", config_module)
    upstream_name = "vllm.v1.worker.gpu.spec_decode.leagle.speculator"
    upstream = ModuleType(upstream_name)

    class LEagleSpeculator:
        def __init__(self, config, device):
            self.config, self.device = config, device

    upstream.LEagleSpeculator = LEagleSpeculator
    monkeypatch.setitem(sys.modules, upstream_name, upstream)
    graph_name = "vllm.v1.worker.gpu.spec_decode.leagle.graph"
    graph_module = ModuleType(graph_name)
    graph_module.NativeLEagleGraphBackend = lambda api, device, cls: (api, device, cls)
    monkeypatch.setitem(sys.modules, graph_name, graph_module)
    root = Path(__file__).resolve().parents[2]
    name = "vllm_ascend.worker.v2.spec_decode.leagle"
    spec = importlib.util.spec_from_file_location(name, root / "vllm_ascend/worker/v2/spec_decode/leagle.py")
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    monkeypatch.setitem(sys.modules, name, adapter)
    spec = importlib.util.spec_from_file_location(
        "ascend_speculator_factory",
        root / "vllm_ascend/worker/v2/spec_decode/__init__.py",
    )
    factory = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(factory)
    config = SimpleNamespace(speculative_config=SimpleNamespace(method="leagle"))
    proposer = factory.init_speculator(config, "npu:0")
    assert type(proposer) is adapter.AscendLEagleSpeculator
    assert isinstance(proposer, LEagleSpeculator)
    assert proposer.device == "npu:0"
    npu = SimpleNamespace(NPUGraph=object)
    monkeypatch.setattr(torch, "npu", npu, raising=False)
    assert proposer.create_graph_backend() == (npu, "npu:0", object)
    quant = ModuleType("vllm_ascend.worker.v2.spec_decode.leagle_quant")
    quant.AscendLEagleInt8Head = lambda weight: ("int8", weight)
    monkeypatch.setitem(sys.modules, quant.__name__, quant)
    installed = []
    weight = torch.randn(64, 32)
    proposer.speculative_config = SimpleNamespace(leagle_lm_head_quantization="w8a8")
    proposer.model = SimpleNamespace(
        candidate_head_enabled=True,
        svd_candidate_head=SimpleNamespace(up_weight=weight),
        set_quantized_svd_head=installed.append,
    )
    proposer.configure_lm_head_quantization()
    assert installed == [("int8", weight)]
    proposer.model.svd_candidate_head = None
    with pytest.raises(ValueError, match="requires SVD"):
        proposer.configure_lm_head_quantization()


@torch.inference_mode()
def test_int8_head_preserves_source_and_matches_scaled_integer_reference():
    path = Path(__file__).resolve().parents[2] / "vllm_ascend/worker/v2/spec_decode/leagle_quant.py"
    spec = importlib.util.spec_from_file_location("leagle_quant_cpu_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def dynamic(x, dst_type):
        assert dst_type == torch.int8
        maximum = x.float().abs().amax(-1)
        scale = torch.where(maximum == 0, 1.0, maximum / 127)
        return (x.float() / scale[:, None]).round().clamp(-127, 127).to(torch.int8), scale

    def matmul(x, w, scale, pertoken_scale, output_dtype):
        assert x.dtype == w.dtype == torch.int8
        assert scale.dtype == pertoken_scale.dtype == torch.float32
        return ((x.int() @ w.int()).float() * scale[None, :] * pertoken_scale[:, None]).to(output_dtype)

    ops = SimpleNamespace(npu_dynamic_quant=dynamic, npu_quant_matmul=matmul)
    # Exact integer-valued rows give an independent no-quantization-error case.
    weight = torch.arange(64 * 32).reshape(64, 32).remainder(128).to(torch.bfloat16)
    weight[0].zero_()
    original = weight.clone()
    head = module.AscendLEagleInt8Head(weight, ops=ops)
    assert head.weight_int8.shape == (32, 64)
    assert head.weight_int8.dtype == torch.int8
    assert head.weight_scale[0] == 1
    assert not head.state_dict() and not list(head.parameters())
    torch.testing.assert_close(weight, original)
    hidden = torch.arange(64).reshape(2, 1, 32).to(torch.bfloat16)
    q, scale = dynamic(hidden.flatten(0, 1), torch.int8)
    expected = matmul(q, head.weight_int8, head.weight_scale, scale, torch.bfloat16)
    torch.testing.assert_close(head(hidden).view(2, 64), expected)
    assert torch.isfinite(head(torch.zeros_like(hidden))).all()
    assert head(hidden[:0]).shape == (0, 1, 64)
    dequantized = head.weight_int8.T.float() * head.weight_scale[:, None]
    assert ((dequantized - original.float()).abs() <= head.weight_scale[:, None] * 0.501).all()
    with pytest.raises(ValueError, match="BF16"):
        head(hidden.float())
    with pytest.raises(ValueError, match="Ascend"):
        module.AscendLEagleInt8Head(weight)
    invalid = weight.clone()
    invalid[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        module.AscendLEagleInt8Head(invalid, ops=ops)
    with pytest.raises(ValueError, match="multiples of 32"):
        module.AscendLEagleInt8Head(weight[:, :31], ops=ops)
