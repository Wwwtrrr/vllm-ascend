# SPDX-License-Identifier: Apache-2.0
"""CPU regression coverage for MMuse ACL update ordering and metadata lifetime."""

import ast
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

SOURCE = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/v2/spec_decode/mmuse/greedy.py"


def load_class(class_name, base, **scope):
    tree = ast.parse(SOURCE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    namespace = {"MMuseGraphManager": base, "MMuseGreedySpeculator": base, **scope}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace[class_name]


def test_replay_resolves_current_metadata_once_and_updates_after_submission():
    events = []

    class Base:
        def run_fullgraph(self, desc):
            events.append("replay")

    class Graph:
        def resolve_tasks(self, context):
            events.append(("resolve", context))
            return context

        def update(self, stream, tasks):
            events.append(("update", tasks))

    cls = load_class(
        "MMuseAclGraphManager",
        Base,
        UpdatableGraph=Graph,
        ContextSource=lambda x: x,
        torch=NS(npu=NS(current_stream=lambda: "current")),
    )
    manager = cls()
    manager.graphs = {"shape": Graph()}
    manager.proposer = NS(
        replay_metadata={"step": 1}, update_stream=NS(wait_stream=lambda stream: events.append(("wait", stream)))
    )
    manager.run_fullgraph("shape")
    manager.proposer.replay_metadata = {"step": 2}
    manager.run_fullgraph("shape")
    assert events == [
        ("resolve", {"step": 1}),
        ("wait", "current"),
        "replay",
        ("update", {"step": 1}),
        ("resolve", {"step": 2}),
        ("wait", "current"),
        "replay",
        ("update", {"step": 2}),
    ]


def test_capture_context_restores_flags_after_failure_and_pads_queries():
    flags = NS(capturing=False, is_draft_model=False, is_draft_model_prefill=True)
    cls = load_class(
        "AscendMMuseGreedySpeculator",
        object,
        MMuseAclGraphManager=object,
        build_attn_metadata=lambda: None,
        contextmanager=contextmanager,
        _EXTRA_CTX=flags,
        torch=NS(npu=NS(is_current_stream_capturing=lambda: True)),
    )
    proposer = cls()
    with pytest.raises(RuntimeError), proposer.query_context():
        assert flags.capturing and flags.is_draft_model
        assert not flags.is_draft_model_prefill
        raise RuntimeError("capture failed")
    assert vars(flags) == dict(capturing=False, is_draft_model=False, is_draft_model_prefill=True)
    proposer.num_query_per_req = 7
    metadata = {"layer": NS(actual_seq_lengths_q=[7, 7])}
    proposer.fix_query_lengths(metadata, 2)
    assert metadata["layer"].actual_seq_lengths_q == [7, 14]
