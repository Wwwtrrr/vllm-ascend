# SPDX-License-Identifier: Apache-2.0
"""CPU coverage of the Target-only Qwen3 FULL specialization gate."""

import ast
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


@pytest.mark.parametrize(
    "model_type,mode,varlen,expected",
    [
        ("qwen3", "FULL", False, True),
        ("qwen3", "NONE", False, False),
        ("qwen3", "FULL_DECODE_ONLY", False, False),
        ("qwen3", "FULL", True, False),
        ("deepseek_v4", "FULL", False, False),
    ],
)
def test_specialization_is_limited_to_qwen3_fixed_length_full(model_type, mode, varlen, expected):
    path = Path(__file__).resolve().parents[4] / "vllm_ascend/worker/v2/aclgraph_utils.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ModelAclGraphManager")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")

    class Parent:
        def __init__(self, *args, **kwargs):
            # This must be set before the parent builds its capture candidates.
            assert self.specialize_full_decode is expected
            self._capture_descs = {}

        def needs_capture(self):
            return False

    minimal = ast.ClassDef(
        name="Manager",
        bases=[ast.Name(id="Parent", ctx=ast.Load())],
        keywords=[],
        body=[init],
        decorator_list=[],
    )
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), minimal],
        type_ignores=[],
    )
    namespace = {
        "Parent": Parent,
        "CUDAGraphMode": NS(FULL="FULL"),
        "collect_sorted_captured_token_sizes": lambda descs: [],
    }
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    config = NS(model_config=NS(hf_config=NS(model_type=model_type)))
    manager = namespace["Manager"](config, None, mode, 8, NS(update_stream=None), varlen_decode=varlen)
    assert manager.specialize_full_decode is expected
