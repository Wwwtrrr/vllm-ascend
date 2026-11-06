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


@torch.inference_mode()
def test_target_input_graph_replay_uses_new_slots_lengths_and_tokens(monkeypatch):
    base = ModuleType("vllm.v1.worker.gpu.cudagraph_utils")
    base.CudaGraphManager = type("CudaGraphManager", (), {})
    monkeypatch.setitem(sys.modules, base.__name__, base)
    path = Path(__file__).resolve().parents[2] / "vllm_ascend/worker/v2/spec_decode/leagle_inputs.py"
    spec = importlib.util.spec_from_file_location("leagle_target_inputs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    buffers = SimpleNamespace(
        input_ids=torch.full((16,), -1),
        positions=torch.full((16,), -1),
        seq_lens=torch.zeros(3, dtype=torch.int32),
        query_start_loc=torch.zeros(5, dtype=torch.int32),
    )
    requests = SimpleNamespace(
        num_computed_tokens=SimpleNamespace(gpu=torch.tensor([8, 12, 5])),
        prefill_len=SimpleNamespace(gpu=torch.tensor([2, 3, 1]), np=[2, 3, 1]),
        last_sampled_tokens=torch.tensor([101, 102, 103]),
        draft_tokens=torch.tensor([[201, 202], [211, 212], [221, 222]]),
    )

    def expand(slots, capacity, cu, query_len):
        expanded = torch.zeros(capacity, dtype=torch.int32)
        local = torch.zeros_like(expanded)
        for i, slot in enumerate(slots.tolist()):
            begin, end = int(cu[i]), int(cu[i + 1])
            expanded[begin:end] = slot
            local[begin:end] = torch.arange(end - begin)
        return expanded, local

    def positions(slots, query, computed, pos, seq):
        seq.zero_()
        for i, slot in enumerate(slots.tolist()):
            begin, end = int(query[i]), int(query[i + 1])
            pos[begin:end] = computed[slot] + torch.arange(end - begin)
            seq[i] = computed[slot] + end - begin

    def combine(ids, slots, sampled, query, seq, prefill, draft, cu, capacity, bonus):
        indices = torch.zeros(capacity, dtype=torch.long)
        for i, slot in enumerate(slots.tolist()):
            begin, end = int(cu[i]), int(cu[i + 1])
            query_end = int(query[i + 1])
            start = query_end - (end - begin)
            indices[begin:end] = torch.arange(start, query_end)
            ids[start] = sampled[slot]
            ids[start + 1 : query_end] = draft[slot, : end - begin - 1]
        return indices

    uploads = []

    def upload(data, out):
        uploads.append(1)
        out.copy_(torch.from_numpy(data))

    class Backend:
        def __init__(self):
            self.captures = 0

        def capture(self, forward):
            self.captures += 1
            from torch.utils._pytree import tree_map

            outputs = tree_map(lambda t: t.clone(), forward())

            def replay():
                tree_map(lambda target, value: target.copy_(value), outputs, forward())

            return SimpleNamespace(replay=replay), outputs

    state = module.LEagleTargetInputState(buffers, requests, 3, 3, functions=(expand, positions, combine), h2d=upload)

    class Tables:
        num_kv_cache_groups = 1
        num_blocks = SimpleNamespace(np=torch.tensor([[4, 4, 4]]).numpy(), gpu=torch.tensor([[4, 4, 4]]))
        rows = torch.arange(12).view(3, 4)

        def gather_block_tables(self, slots, num_reqs_padded, num_blocks=None):
            result = torch.zeros(num_reqs_padded, 4, dtype=torch.long)
            for row, slot in enumerate(slots.tolist()):
                size = int(num_blocks[0, slot])
                result[row, :size] = self.rows[slot, :size]
            return (result,)

        def compute_slot_mappings(self, slots, query, positions, num_tokens_padded):
            result = torch.full((1, num_tokens_padded), -1)
            for i, slot in enumerate(slots.tolist()):
                start, end = int(query[i]), int(query[i + 1])
                pos = positions[start:end]
                result[0, start:end] = self.rows[slot, pos // 4] * 4 + pos % 4
            return result

    state.block_tables = Tables()
    backend = Backend()
    manager = module.LEagleTargetInputGraphRunner(state, backend)
    with pytest.raises(RuntimeError, match="not captured"):
        manager.replay(2, 4)
    manager.capture()
    assert backend.captures == 3
    # FIA may leave unused tail boundaries below the active padding boundary.
    state.stage([2, 0], [0, 3, 4], [0, 3, 4, 8, 4])
    before = len(uploads)
    expanded, local, logits = manager.replay(2, 4)
    assert len(uploads) == before
    assert expanded.tolist() == [2, 2, 2, 0]
    assert local.tolist() == [0, 1, 2, 0]
    assert logits.tolist() == [0, 1, 2, 3]
    assert buffers.input_ids[:4].tolist() == [103, 221, 222, 101]
    assert buffers.positions[:4].tolist() == [5, 6, 7, 8]
    assert buffers.seq_lens.tolist() == [8, 9, 0]
    tables, mappings = manager.prepared_attn
    torch.testing.assert_close(tables[0][0], state.block_tables.rows[2])
    assert (tables[0][2] == 0).all()
    assert (mappings[0, 4:] == -1).all()
    # The original UVA-style pointer changes; replay must use freshly staged
    # counts rather than the pointer present during capture.
    state.block_tables.num_blocks.np[0, 2] = 3
    state.block_tables.num_blocks.gpu = torch.tensor([[4, 4, 3]])
    requests.num_computed_tokens.gpu.copy_(torch.tensor([10, 12, 7]))
    state.stage([0, 2], [0, 2, 5], [0, 2, 5, 8, 5])
    expanded, local, logits = manager.replay(2, 5)
    assert expanded.tolist() == [0, 0, 2, 2, 2]
    assert local.tolist() == [0, 1, 0, 1, 2]
    assert buffers.input_ids[:5].tolist() == [101, 201, 103, 221, 222]
    assert buffers.positions[:5].tolist() == [10, 11, 7, 8, 9]
    assert buffers.seq_lens.tolist() == [12, 10, 0]
    tables, mappings = manager.prepared_attn
    assert tables[0][1, 3] == 0
    assert mappings[0, :5].tolist() == [10, 11, 39, 40, 41]
    assert manager.replays == 2 and backend.captures == 3
    with pytest.raises(ValueError, match="differs"):
        manager.replay(2, 4)
    with pytest.raises(ValueError, match="decode-only"):
        state.stage([0], [0, 1], [0, 2, 2, 2, 2])
    with pytest.raises(ValueError, match="real request slots"):
        state.stage([-1], [0, 1], [0, 1, 1, 1, 1])
    with pytest.raises(ValueError, match="captured decode length"):
        state.stage([0], [0, 4], [0, 4, 4, 4, 4])
    # Profiling teardown drops only graphs, retaining stable metadata storage.
    metadata = state.metadata.data_ptr()
    manager = module.LEagleTargetInputGraphRunner(state, Backend())
    manager.capture()
    state.stage([1], [0, 1], [0, 1, 1, 1, 1])
    manager.replay(1, 1)
    assert buffers.input_ids[0] == 102
    assert state.metadata.data_ptr() == metadata
