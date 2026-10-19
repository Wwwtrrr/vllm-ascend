# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for actual ACL draft replay methods, with device calls mocked.

The AST loader avoids importing torch_npu on CPU-only development machines.
These tests validate metadata/config handoff, not ACL capture or kernel results.
"""

import __future__

import ast
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def test_capture_wrapper_forwards_mmuse_rollout_without_generic_getattr():
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/v2/aclgraph_utils.py"
    tree = ast.parse(path.read_text())
    wrapper = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ModelWithContext")
    method = next(node for node in wrapper.body if isinstance(node, ast.FunctionDef) and node.name == "rollout_mmuse")
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    original = Mock()
    obj = SimpleNamespace(original_model=original)
    hidden, anchor, sampler = object(), object(), object()
    result = namespace["rollout_mmuse"](obj, hidden, anchor, sample_token=sampler)
    original.rollout_mmuse.assert_called_once_with(hidden, anchor, sample_token=sampler)
    assert result is original.rollout_mmuse.return_value


@pytest.fixture
def replay_class():
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/v2/spec_decode/dflash/aclgraph.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DFlashAclGraphManager")
    cls.bases = [ast.Name(id="Parent", ctx=ast.Load())]
    cls.body = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name in {"run_fullgraph", "_run_fullgraph_with_config"}
    ]
    active = []

    @contextmanager
    def current(config):
        active.append(config)
        try:
            yield
        finally:
            active.pop()

    def updatable(backend):
        assert active
        return backend == "updatable"

    namespace = {
        "Parent": object,
        "set_current_vllm_config": current,
        "use_updatable_graph": updatable,
    }
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])),
            str(path),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )
    return namespace["DFlashAclGraphManager"], active


@pytest.mark.parametrize("prepared", [True, False])
@pytest.mark.parametrize("updatable", [True, False])
@pytest.mark.parametrize("fail", [True, False])
def test_acl_replay_reuses_metadata_and_restores_current_config(replay_class, prepared, updatable, fail):
    cls, active = replay_class
    obj = cls()
    obj.vllm_config = object()
    metadata = {"draft_layer": object()}
    speculator = SimpleNamespace(
        attn_backends={"draft_layer": "updatable" if updatable else "legacy"},
        input_batch=SimpleNamespace(seq_lens_cpu_upper_bound=object()),
        build_draft_attn_metadatas=Mock(return_value=[metadata]),
        _update_draft_attn_metadata=Mock(),
        _mmuse_replay_attn_metadata=metadata if prepared else None,
    )
    obj.speculator = speculator

    def replay(*args):
        assert active == [obj.vllm_config]
        assert args[-1][0] is metadata
        if fail:
            raise RuntimeError("replay failed")
        return "submitted"

    obj._graph_replay = Mock(side_effect=replay)
    obj._updatable_graph_replay = Mock(side_effect=replay)
    desc = SimpleNamespace(num_tokens=28, num_reqs=4)
    if fail:
        with pytest.raises(RuntimeError, match="replay failed"):
            obj.run_fullgraph(desc)
    else:
        assert obj.run_fullgraph(desc) == "submitted"
    assert not active
    if prepared:
        speculator.build_draft_attn_metadatas.assert_not_called()
        speculator._update_draft_attn_metadata.assert_called_once_with(metadata, 4)
    else:
        speculator.build_draft_attn_metadatas.assert_called_once()
        speculator._update_draft_attn_metadata.assert_not_called()
    assert obj._updatable_graph_replay.call_count == int(updatable)
    assert obj._graph_replay.call_count == int(not updatable)
