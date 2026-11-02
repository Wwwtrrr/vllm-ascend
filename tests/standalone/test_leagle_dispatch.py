# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check Ascend LEAGLE routing without requiring an NPU runtime."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace


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
