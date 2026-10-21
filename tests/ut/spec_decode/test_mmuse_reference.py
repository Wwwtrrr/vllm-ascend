# SPDX-License-Identifier: Apache-2.0
"""CPU test of the reference proposer's NPU sampling binding."""

import ast
import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS

import torch


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


def test_lmhead_tp_trace_receives_trimmed_logits_before_verification():
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/v2/model_runner.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "sample")
    namespace = dict(lmhead_tp_enable=lambda: True, lmhead_tp_pad_rows=lambda *args: torch.tensor([0, 1, 0, 0]))
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    events = []
    batch = NS(logits_indices=torch.tensor([0, 1]), num_draft_tokens=1)
    output = NS(num_sampled=torch.tensor([1]), num_rejected=torch.tensor([1]))

    def verify(logits, actual_batch, draft_logits):
        events.append(("verify", tuple(logits.shape)))
        assert actual_batch is batch
        return output

    runner = NS(
        _lmhead_tp_max_num_logits=lambda: 4,
        model=NS(compute_logits=lambda hidden: torch.zeros(hidden.shape[0], 8)),
        speculator=NS(draft_logits=None, trace_target=lambda logits, _: events.append(("trace", tuple(logits.shape)))),
        rejection_sampler=verify,
    )
    result = namespace["sample"](runner, torch.zeros(4, 3), batch, None)
    assert events == [("trace", (2, 8)), ("verify", (2, 8))]
    assert result[0] is output
