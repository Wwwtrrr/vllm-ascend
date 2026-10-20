# SPDX-License-Identifier: Apache-2.0
"""CPU layout contracts for DSV4 + dense MMuse, without importing NPU code."""

import __future__

import ast
import copy
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


@dataclass
class MLA:
    page_size_bytes: int = 10


@dataclass
class SWAMLA:
    page_size_bytes: int = 10


@dataclass
class Dense:
    page_size_bytes: int = 16


@dataclass
class Sliding(Dense):
    pass


@dataclass
class Uniform:
    kv_cache_specs: dict


@dataclass
class Tensor:
    size: int
    layers: list
    offset: int = 0
    layer_stride: int = 0
    block_stride: int = 0


@pytest.fixture
def planner():
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/patch/platform/patch_kv_cache_utils.py"
    tree = ast.parse(path.read_text())
    names = {
        "_split_dsv4_dense_groups",
        "_dense_cache_layers",
        "_dsv4_dense_cache_config",
        "_ascend_get_packed_kv_cache_groups",
    }
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]

    def native(config, groups, memory):
        blocks = config.cache_config.num_gpu_blocks_override
        return blocks, [
            Tensor(blocks * 100, ["target"], block_stride=10),
            Tensor(blocks * 100, ["swa_target"], block_stride=10),
        ]

    namespace = {
        "copy": copy,
        "replace": replace,
        "UniformTypeKVCacheSpecs": Uniform,
        "MLAAttentionSpec": MLA,
        "SlidingWindowMLASpec": SWAMLA,
        "FullAttentionSpec": Dense,
        "SlidingWindowSpec": Sliding,
        "KVCacheTensor": Tensor,
        "KVCacheConfig": lambda **kwargs: NS(**kwargs),
        "_is_deepseek_v4_groups": lambda groups: bool(groups),
        "_ascend_pool_bytes_per_block": lambda groups: 100,
        "_get_kv_cache_config_deepseek_v4_main": native,
        "may_override_num_blocks": lambda cfg, blocks: cfg.cache_config.num_gpu_blocks_override or blocks,
        "is_deepseek_v41_cache": lambda specs: False,
        "group_and_unify_kv_cache_specs": lambda specs: (
            [NS(layer_names=list(specs), kv_cache_spec=Uniform(specs))]
            if any(isinstance(spec, MLA) for spec in specs.values())
            else None
        ),
        "_get_kv_cache_groups_uniform_groups": lambda groups: groups,
        "_orig_get_kv_cache_groups": lambda cfg, specs: [NS(layer_names=list(specs), kv_cache_spec=Uniform(specs))],
        "_orig_get_packed_kv_cache_groups": lambda cfg, specs: [
            NS(layer_names=list(specs), kv_cache_spec=Uniform(specs))
        ],
        "vllm_version_is": lambda version: False,
        "vllm": NS(
            v1=NS(
                core=NS(
                    kv_cache_utils=NS(
                        _annotate_eagle_groups=lambda *args, **kwargs: None,
                        _warn_if_unannotated_eagle_mamba=lambda *args, **kwargs: None,
                    )
                )
            )
        ),
    }
    exec(
        compile(
            ast.Module(body=functions, type_ignores=[]), str(path), "exec", flags=__future__.annotations.compiler_flag
        ),
        namespace,
    )
    return namespace


def test_packed_hook_keeps_every_dense_draft_and_native_target_layer(planner):
    cfg = NS(speculative_config=NS(method="mmuse"), model_config=NS(hf_config=NS(model_type="deepseek_v4")))
    specs = {"target": MLA(), "swa_target": SWAMLA(), "draft0": Dense(), "draft1": Sliding()}
    planned = planner["_ascend_get_packed_kv_cache_groups"](cfg, specs)
    names = [name for group in planned for name in group.layer_names]
    assert sorted(names) == sorted(specs)
    assert len(names) == len(set(names))
    assert planned[0].is_eagle_group is False
    assert planned[1].is_eagle_group is True


def groups():
    return [
        NS(layer_names=["target"], kv_cache_spec=Uniform({"target": MLA()})),
        NS(layer_names=["swa_target"], kv_cache_spec=Uniform({"swa_target": SWAMLA()})),
        NS(layer_names=["draft0", "draft1"], kv_cache_spec=Uniform({"draft0": Dense(16), "draft1": Sliding(24)})),
    ]


def test_mla_and_swa_mla_stay_native_and_dense_layers_are_not_dropped(planner):
    original = groups()
    native, dense = planner["_split_dsv4_dense_groups"](original)
    assert native == original[:2]
    assert list(planner["_dense_cache_layers"](dense)) == [("draft0", Dense(16)), ("draft1", Sliding(24))]
    assert planner["_split_dsv4_dense_groups"](original[:2]) is None


@pytest.mark.parametrize("override", [None, 5])
def test_one_backing_preserves_native_offsets_and_reserves_dense_regions(planner, override):
    cfg = NS(cache_config=NS(num_gpu_blocks_override=override, prefix_cache_retention_interval=None))
    result = planner["_dsv4_dense_cache_config"](cfg, groups(), 1400)
    blocks = override or 10
    assert result.num_blocks == blocks
    assert cfg.cache_config.num_gpu_blocks_override == override
    assert {tensor.size for tensor in result.kv_cache_tensors} == {blocks * 140}
    target, swa_target, first, second = result.kv_cache_tensors
    assert (target.offset, target.block_stride) == (0, 10)
    assert (swa_target.offset, swa_target.block_stride) == (0, 10)
    assert first.offset == blocks * 100
    assert second.offset == blocks * 116
    assert second.offset + blocks * second.block_stride == second.size
    assert [first.layers, second.layers] == [["draft0"], ["draft1"]]
