# SPDX-License-Identifier: Apache-2.0
"""DeepSeek native graph backend and teacher-space binding contracts."""

import ast
import importlib.util
import sys
from contextlib import nullcontext
from dataclasses import dataclass, replace
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


def attention_module():
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/v2/spec_decode/mmuse/attention.py"
    spec = importlib.util.spec_from_file_location("mmuse_fia_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def attention_bridge():
    return attention_module().fused_context_attention


def test_framework_binding_registers_cache_owners_without_target_quantizer(monkeypatch):
    module = attention_module()
    config = NS(
        kv_transfer_config=None, cache_config=NS(cache_dtype="fp8", sliding_window=32, calculate_kv_scales=True)
    )
    registered = {}
    config_module = ModuleType("vllm.config")
    config_module.set_current_vllm_config = lambda cfg: nullcontext(cfg)
    owners = ModuleType("vllm.model_executor.layers.attention")

    class Owner(torch.nn.Module):
        def __init__(self, heads, dim, scale, **kwargs):
            super().__init__()
            self.layer_name = kwargs["prefix"]
            self.num_kv_heads, self.head_size = kwargs["num_kv_heads"], dim
            self.kv_cache = torch.empty(0)
            assert kwargs["quant_config"] is None
            assert kwargs["cache_config"].cache_dtype == "auto"
            assert kwargs["cache_config"].sliding_window is None
            assert not kwargs["cache_config"].calculate_kv_scales
            registered[self.layer_name] = self

    owners.Attention = Owner
    backend_module = ModuleType("vllm_ascend.attention.attention_v1")
    backend_module.AscendAttentionBackend = object()
    for item in (config_module, owners, backend_module):
        monkeypatch.setitem(sys.modules, item.__name__, item)
    layers = [NS(self_attn=NS(heads=4, kv_heads=2, dim=64, paged_attention_impl=None)) for _ in range(2)]
    model = NS(layers=layers, feature_storage=torch.empty(1, 256, 128, dtype=torch.bfloat16))
    module.bind_framework_attention(model, config, 7)
    assert len(registered) == 2
    for index, layer in enumerate(layers):
        paged = layer.self_attn.paged_attention_impl
        assert paged.cache_owner is registered[f"draft_model.layers.{index}.self_attn.attn"]
        assert "key_cache" not in paged._buffers  # No private K/V arena.
    assert config.cache_config.cache_dtype == "fp8"
    assert config.cache_config.sliding_window == 32
    config.kv_transfer_config = object()
    with pytest.raises(ValueError, match="KV transfer"):
        module.bind_framework_attention(model, config, 7)


@pytest.mark.parametrize("device", ["cpu", "npu"])
@torch.inference_mode()
def test_framework_pages_follow_absolute_positions_and_reallocated_tables(monkeypatch, device):
    if device == "npu":
        pytest.importorskip("torch_npu")
        if not torch.npu.is_available():
            pytest.skip("NPU required")
    else:
        install_paged_oracle(monkeypatch)
    torch.manual_seed(121)
    owner = NS(
        kv_cache=torch.full((2, 9, 128, 2, 64), float("nan"), dtype=torch.bfloat16, device=device),
        num_kv_heads=2,
        head_size=64,
    )
    arena = attention_module().FrameworkContextAttention(
        owner, torch.empty(1, 256, 128, dtype=owner.kv_cache.dtype, device=device), 7
    )
    arena.prepare_for_capture()
    q = torch.randn(7, 4, 64, dtype=owner.kv_cache.dtype, device=device)
    qk = torch.randn(7, 2, 64, dtype=q.dtype, device=device)
    qv = torch.randn_like(qk)
    for first, end, pages in [
        (0, 125, [3, 1]),
        (0, 132, [3, 1]),
        (125, 257, [4, 6, 2]),
        (129, 260, [4, 6, 2]),
        (0, 4, [6]),
        (0, 125, [1, 3]),
    ]:
        table = torch.zeros(8, dtype=torch.int32, device=device)
        table[: len(pages)] = table.new_tensor(pages)
        arena.begin_request(table, first, end)
        length = end - first
        key = torch.randn(256, 2, 64, dtype=q.dtype, device=device)
        value = torch.randn_like(key)
        key[length:] = value[length:] = float("nan")
        arena.prepare_context(key, value, start=min(3, length))
        visible = torch.ones(7, 263, dtype=torch.bool, device=device)
        visible[:, :2] = False
        visible[:, length:256] = False
        visible[:, 256:] = torch.ones(7, 7, dtype=torch.bool, device=device).tril()
        compact_mask = torch.cat((visible[:, :length], visible[:, 256:]), -1)
        expected = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(0, 1)[None],
            torch.cat((key[:length], qk)).transpose(0, 1)[None].repeat_interleave(2, 1),
            torch.cat((value[:length], qv)).transpose(0, 1)[None].repeat_interleave(2, 1),
            attn_mask=compact_mask,
        )[0].transpose(0, 1)
        actual = arena(q, qk, qv, visible)
        torch.testing.assert_close(actual, expected, atol=0.03, rtol=0.03)
    # Capture uses reserved block zero and inert writes; actual replay changes
    # table, span and context on stable tensor addresses, not graph arguments.
    if device == "npu":
        arena.prepare_for_capture()
        for _ in range(3):
            arena(q, qk, qv, visible)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            output = arena(q, qk, qv, visible)
        arena.begin_request(table, first, end)
        arena.prepare_context(key, value)
        expected = arena(q, qk, qv, visible)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(output, expected, atol=0.03, rtol=0.03)


@torch.inference_mode()
def test_framework_short_allocation_falls_back_without_page_writes(monkeypatch):
    writes = install_paged_oracle(monkeypatch)
    module = attention_module()
    owner = NS(kv_cache=torch.zeros(2, 4, 128, 2, 64, dtype=torch.bfloat16), num_kv_heads=2, head_size=64)
    arena = module.FrameworkContextAttention(owner, torch.empty(1, 256, 128), 7)
    arena.begin_request(torch.tensor([2], dtype=torch.int32), 0, 125)
    assert arena.requires_eager
    assert not writes
    context = torch.randn(125, 2, 64)
    query = torch.randn(7, 4, 64)
    key, value = torch.randn(7, 2, 64), torch.randn(7, 2, 64)
    visible = torch.ones(7, 132, dtype=torch.bool)
    seen = []

    def dense(q, k, v, mask):
        seen.append((k.shape, v.shape, mask))
        return q

    monkeypatch.setattr(module, "fused_context_attention", dense)
    arena.prepare_context(context, context)
    torch.testing.assert_close(arena(query, key, value, visible), query)
    assert seen[0][:2] == (torch.Size([132, 2, 64]), torch.Size([132, 2, 64]))
    assert not writes
    assert arena._profile_context is None
    arena.begin_request(torch.tensor([2, 3], dtype=torch.int32), 0, 125)
    assert not arena.requires_eager


def install_paged_oracle(monkeypatch):
    """Emulate native page writes and PA reads; no device-kernel claim."""
    device = ModuleType("vllm_ascend.device.device_op")
    writes = []

    def write(key, value, key_cache, value_cache, slots):
        writes.append(slots.clone())
        valid = slots >= 0
        key_cache.flatten(0, 1).index_copy_(0, slots[valid].long(), key[valid])
        value_cache.flatten(0, 1).index_copy_(0, slots[valid].long(), value[valid])

    device.DeviceOperator = NS(reshape_and_cache=write)
    monkeypatch.setitem(sys.modules, device.__name__, device)
    npu = ModuleType("torch_npu")

    def fia(**kwargs):
        table = kwargs["block_table"][0].long()
        length = kwargs["actual_seq_lengths_kv"][0]
        heads = kwargs["num_key_value_heads"]
        dim = kwargs["query"].shape[-1]
        keys = kwargs["key"][table].reshape(-1, heads, dim)[:length]
        values = kwargs["value"][table].reshape(-1, heads, dim)[:length]
        groups = kwargs["num_heads"] // heads
        return torch.nn.functional.scaled_dot_product_attention(
            kwargs["query"],
            keys.transpose(0, 1)[None].repeat_interleave(groups, 1),
            values.transpose(0, 1)[None].repeat_interleave(groups, 1),
            attn_mask=~kwargs["atten_mask"],
        ), None

    npu.npu_fused_infer_attention_score = fia
    monkeypatch.setitem(sys.modules, npu.__name__, npu)
    return writes


@torch.inference_mode()
def test_paged_decoder_and_rollout_match_reference_through_rejection_and_reuse(monkeypatch):
    path = Path(__file__).resolve().parents[4] / "vllm/tests/v1/spec_decode/test_mmuse_deepseek_graph.py"
    spec = importlib.util.spec_from_file_location("mmuse_paged_decoder_test", path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    torch.manual_seed(101)
    owner = helper.model("cpu", torch.bfloat16, (4, None), monkeypatch)
    install_paged_oracle(monkeypatch)
    paged_cls = attention_module().PagedContextAttention
    pages = [paged_cls(owner.feature_storage, 2, 4, 3) for _ in owner.layers]
    graph_cls = helper.pure_module("deepseek_graph").DeepseekMMuseGraphs
    args = (owner, 3, None, None, torch.zeros(2), torch.zeros(2, dtype=torch.long), False, None)
    driver, reference = graph_cls(*args), graph_cls(*args)
    features = torch.randn(30, 16, dtype=torch.bfloat16)
    rounds = [
        (0, "a", 0, 5, 0),
        (0, "a", 0, 12, 5),
        (0, "a", 2, 18, 12),
        (0, "a", 2, 18, 14),
        (0, "a", 2, 8, 8),
        (1, "b", 0, 3, 0),
        (0, "new", 0, 6, 0),
    ]
    for slot, request, start, end, update in rounds:
        features[update:end] += 0.1
        positions, context = torch.arange(start, end), features[start:end]
        anchor = torch.tensor([7])
        for layer in owner.layers:
            layer.self_attn.paged_attention_impl = None
        size = reference.prepare(positions, context, anchor, slot)
        reference.forward(size)
        expected_tokens = reference.output.clone()
        model_args = (
            reference.input_ids,
            reference.positions,
            reference.context_positions[:size],
            reference.context[:size],
            tuple(kv[:, :size] for kv in reference.context_kv),
            reference.anchor_index,
        )
        expected_hidden = owner.forward_fixed_context(*model_args)
        _, expected_logits = owner.runtime.rollout_correction(expected_hidden[None], anchor)
        # Exercise eager loading/profile fallback as well as prepared graph context.
        expected_eager = owner(anchor.new_tensor([7, 31, 31]), torch.arange(end, end + 3), positions, context)
        for layer, page in zip(owner.layers, pages):
            layer.self_attn.paged_attention_impl = page
        actual_eager = owner(anchor.new_tensor([7, 31, 31]), torch.arange(end, end + 3), positions, context)
        torch.testing.assert_close(actual_eager, expected_eager, atol=0.03, rtol=0.03)
        # Eager/profile calls reuse the arena, so simulate graph lifecycle reset.
        driver.context_entry = None
        size = driver.prepare(
            positions, context, anchor, slot, request_id=request, start=start, end=end, update_start=update
        )
        args = (
            driver.input_ids,
            driver.positions,
            driver.context_positions[:size],
            driver.context[:size],
            tuple(kv[:, :size] for kv in driver.context_kv),
            driver.anchor_index,
        )
        actual_hidden = owner.forward_fixed_context(*args)
        actual_tokens, actual_logits = owner.runtime.rollout_correction(actual_hidden[None], anchor)
        torch.testing.assert_close(actual_hidden, expected_hidden, atol=0.03, rtol=0.03)
        torch.testing.assert_close(actual_logits, expected_logits, atol=0.03, rtol=0.03)
        torch.testing.assert_close(actual_tokens[0], expected_tokens)
    # Consecutive FULL steps reuse pages. Window shift must rewrite the retained
    # prefix because its logical page offsets have changed.
    for start, end, update in [(0, 9, 0), (0, 12, 9), (2, 18, 12), (2, 18, 14)]:
        features[update:end] += 0.1
        size = driver.prepare(
            torch.arange(start, end),
            features[start:end],
            anchor,
            0,
            request_id="continuous",
            start=start,
            end=end,
            update_start=update,
        )
        args = (
            driver.input_ids,
            driver.positions,
            driver.context_positions[:size],
            driver.context[:size],
            tuple(kv[:, :size] for kv in driver.context_kv),
            driver.anchor_index,
        )
        actual = owner.forward_fixed_context(*args)
        for layer in owner.layers:
            layer.self_attn.paged_attention_impl = None
        expected = owner.forward_fixed_context(*args)
        torch.testing.assert_close(actual, expected, atol=0.03, rtol=0.03)
        for layer, page in zip(owner.layers, pages):
            layer.self_attn.paged_attention_impl = page


@pytest.mark.parametrize("device", ["cpu", "npu"])
@torch.inference_mode()
def test_paged_cache_handles_page_crossing_overwrite_and_shorter_context(monkeypatch, device):
    if device == "npu":
        pytest.importorskip("torch_npu")
        if not torch.npu.is_available():
            pytest.skip("NPU required")
        writes = None
    else:
        writes = install_paged_oracle(monkeypatch)
    torch.manual_seed(89)
    arena = attention_module().PagedContextAttention(
        torch.zeros(1, 256, 128, dtype=torch.bfloat16, device=device), 2, 64, 7
    )
    # A nonidentity page table ensures tests exercise block mapping, not a
    # coincidentally contiguous flatten of physical storage.
    arena.block_table.copy_(arena.block_table.flip(-1))
    logical = torch.arange(arena.slots.numel(), device=device)
    arena.slots.copy_(arena.block_table[0, logical // 128].long() * 128 + logical % 128)
    k = torch.randn(263, 2, 64, dtype=torch.bfloat16, device=device)
    v = torch.randn_like(k)
    q = torch.randn(7, 4, 64, dtype=k.dtype, device=device)
    for length, start in [(125, 0), (132, 125), (132, 120), (4, 0), (256, 0)]:
        k[start:length] += 0.1
        arena.prepare_context(k[:length], v[:length], start=start)
        visible = torch.ones(7, length + 7, dtype=torch.bool, device=device)
        visible[:, : max(0, length - 100)] = False
        visible[:, length:] = torch.ones(7, 7, dtype=torch.bool, device=device).tril()
        expected = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(0, 1)[None],
            k[: length + 7].transpose(0, 1)[None].repeat_interleave(2, 1),
            v[: length + 7].transpose(0, 1)[None].repeat_interleave(2, 1),
            attn_mask=visible,
        )[0].transpose(0, 1)
        actual = arena(q, k[length : length + 7], v[length : length + 7], visible)
        torch.testing.assert_close(actual, expected, atol=0.03, rtol=0.03)
        if writes is not None:
            assert writes[-2].numel() == length - start
            assert writes[-1].numel() == 7
    assert not arena.state_dict()  # Runtime pages must not enter checkpoints.
    if device == "npu":
        graph = torch.npu.NPUGraph()
        for _ in range(3):
            arena(q, k[256:], v[256:], visible)
        torch.npu.synchronize()
        with torch.npu.graph(graph):
            output = arena(q, k[256:], v[256:], visible)
        q.add_(0.1)
        k[:256] += 0.2
        arena.prepare_context(k[:256], v[:256])
        expected = arena(q, k[256:], v[256:], visible)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(output, expected, atol=0.03, rtol=0.03)


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


def test_deepseek_mixed_mmuse_cache_groups_cover_layers_and_budget_shared_backing():
    """Dense Draft pages must survive MLA grouping and rank capacity re-plans."""
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/patch/platform/patch_kv_cache_utils.py"
    tree = ast.parse(path.read_text())
    names = {
        "_ascend_get_packed_kv_cache_groups",
        "_split_mmuse_cache_groups",
        "_mmuse_page_bytes",
        "_ascend_pool_bytes_per_block",
        "_ascend_max_memory_usage_bytes_from_groups",
        "_ascend_get_kv_cache_config_from_groups",
    }

    class Full:
        def __init__(self, page):
            self.page_size_bytes = page

        def max_memory_usage_bytes(self, config):
            return self.page_size_bytes * 4

    class MLA(Full):
        model_version = "deepseek_v4"

    class SWA(MLA):
        pass

    class Uniform:
        def __init__(self, specs):
            self.kv_cache_specs = specs

        @classmethod
        def from_specs(cls, specs):
            return cls(specs)

    @dataclass
    class Tensor:
        size: int
        layers: list
        offset: int
        layer_stride: int
        block_stride: int

    def group(names, spec):
        return NS(layer_names=names, kv_cache_spec=spec)

    def target_grouping(config, specs):
        assert all(isinstance(spec, MLA) for spec in specs.values())
        return [group([name], Uniform({name: spec})) for name, spec in specs.items()]

    def layout(groups):
        # The target's two groups overlay one target-only backing region.
        return [16], [], [], 0, 2

    def target_config(config, groups, memory):
        assert all(isinstance(spec, MLA) for g in groups for spec in g.kv_cache_spec.kv_cache_specs.values())
        count = memory // 32
        tensors = [Tensor(32 * count, g.layer_names, 0, 0, 16) for g in groups]
        return NS(num_blocks=count, kv_cache_tensors=tensors, kv_cache_groups=groups)

    config = NS(
        kv_transfer_config=None,
        speculative_config=NS(method="mmuse"),
        cache_config=NS(prefix_cache_retention_interval=1, num_gpu_blocks_override=None),
    )
    scope = dict(
        VllmConfig=object,
        KVCacheSpec=object,
        KVCacheGroupSpec=group,
        FullAttentionSpec=Full,
        MLAAttentionSpec=MLA,
        SlidingWindowMLASpec=SWA,
        UniformTypeKVCacheSpecs=Uniform,
        KVCacheConfig=NS,
        KVCacheTensor=Tensor,
        replace=replace,
        is_deepseek_v41_cache=lambda groups: False,
        group_and_unify_kv_cache_specs=lambda specs: specs,
        _get_kv_cache_groups_uniform_groups=lambda specs: target_grouping(config, specs),
        _is_deepseek_v4_groups=lambda groups: bool(groups),
        _get_deepseek_v4_cache_layout=layout,
        _get_kv_cache_config_deepseek_v4_main=lambda cfg, groups, mem: (
            (result := target_config(cfg, groups, mem)).num_blocks,
            result.kv_cache_tensors,
        ),
        _get_glm5_next_cache_layout=lambda groups: None,
        may_override_num_blocks=lambda cfg, count: cfg.cache_config.num_gpu_blocks_override or count,
        tq_cache=NS(uses_turboquant_groups=lambda groups: False),
        vllm_version_is=lambda version: False,
        vllm=NS(
            v1=NS(
                core=NS(
                    kv_cache_utils=NS(
                        _annotate_eagle_groups=lambda *args, **kwargs: None,
                        _warn_if_unannotated_eagle_mamba=lambda *args: None,
                    )
                )
            )
        ),
    )
    exec(
        compile(
            ast.Module(
                body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[]
            ),
            str(path),
            "exec",
        ),
        scope,
    )
    target = {"model.layers.0.attn": MLA(16), "model.layers.0.swa": SWA(16)}
    draft = {f"draft_model.layers.{i}.self_attn.attn": Full(64) for i in range(2)}
    specs = target | draft
    groups = scope["_ascend_get_packed_kv_cache_groups"](config, specs)
    covered = [name for g in groups for name in g.layer_names]
    assert len(covered) == len(specs) and set(covered) == set(specs)
    divisor = scope["_ascend_pool_bytes_per_block"](groups)
    assert divisor == 32 + 128
    # Override normal target admission formula with its original result: the
    # mixed wrapper must add Draft's memory, without repacking the Target.
    original_memory = scope["_ascend_max_memory_usage_bytes_from_groups"]

    def memory(cfg, current):
        return original_memory(cfg, current) if len(current) > 2 else 128

    scope["_ascend_max_memory_usage_bytes_from_groups"] = memory
    assert memory(config, groups) == 128 + 512
    for budget in [1600, 800]:
        result = scope["_ascend_get_kv_cache_config_from_groups"](config, groups, budget)
        assert result.num_blocks == budget // divisor
        assert {tensor.size for tensor in result.kv_cache_tensors} == {budget}
        assert result.kv_cache_groups is groups
        target_end = result.num_blocks * 32
        spans = [(t.offset, t.offset + result.num_blocks * t.block_stride) for t in result.kv_cache_tensors[2:]]
        assert spans[0][0] == target_end
        assert spans[0][1] == spans[1][0]
        assert spans[1][1] == budget
    config.cache_config.num_gpu_blocks_override = 3
    result = scope["_ascend_get_kv_cache_config_from_groups"](config, groups, 1600)
    assert result.num_blocks == 3
    assert {t.size for t in result.kv_cache_tensors} == {3 * divisor}
