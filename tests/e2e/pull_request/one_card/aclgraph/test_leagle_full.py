# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real NPU capture/replay and target-only/eager/FULL serving equivalence."""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from transformers import Qwen3Config, Qwen3ForCausalLM

pytest.importorskip("torch_npu")
if not torch.npu.is_available():
    pytest.skip("LEAGLE FULL requires an NPU", allow_module_level=True)

from vllm.model_executor.models.qwen3_leagle import LEagleDraftModel
from vllm.v1.worker.gpu.spec_decode.leagle.context import LEagleContext
from vllm.v1.worker.gpu.spec_decode.leagle.execution import (
    LEagleExecution,
    LEagleExecutionOptions,
    LEagleOptimizedGraphRunner,
)
from vllm.v1.worker.gpu.spec_decode.leagle.graph import (
    LEagleGraphInputs,
    LEagleGraphRunner,
    NativeLEagleGraphBackend,
)

from vllm_ascend.worker.v2.spec_decode.leagle import AscendLEagleSpeculator


def _draft(config):
    hf = SimpleNamespace(leagle_config=config)
    return LEagleDraftModel(
        vllm_config=SimpleNamespace(
            speculative_config=SimpleNamespace(draft_model_config=SimpleNamespace(hf_config=hf))
        )
    ).eval()


def _architecture():
    decoder = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=512,
        eos_token_id=None,
        bos_token_id=1,
        pad_token_id=0,
        layer_types=["full_attention", "sliding_attention"],
        sliding_window=16,
    )
    return dict(
        transformer_layer_config=decoder.to_dict(),
        aux_hidden_state_layer_ids=[0, 2, 3],
        num_loops=2,
        norm_output=True,
        draft_vocab_size=64,
    )


@torch.inference_mode()
def test_native_acl_graph_matches_eager_after_context_shrink():
    torch.manual_seed(42)
    model = _draft(_architecture()).to(device="npu", dtype=torch.bfloat16)
    inputs = LEagleGraphInputs.allocate(model, 257, 17, dtype=torch.bfloat16, device="npu")
    manager = LEagleGraphRunner(
        model,
        inputs,
        3,
        NativeLEagleGraphBackend(torch.npu, torch.device("npu:0"), torch.npu.NPUGraph),
    )
    manager.capture()
    for length in (129, 1, 257, 7):
        features = torch.randn(length, 96, device="npu", dtype=torch.bfloat16)
        expected_context = model.project_context(features)
        context = manager.project_context(features)
        torch.testing.assert_close(context, expected_context)
        anchor = torch.tensor([[5]], device="npu")
        expected = model.propose_context(context[None], anchor, 3)
        actual = manager.propose(context[None], anchor, 3)
        torch.testing.assert_close(actual, expected)
    assert manager.proposal_replays == manager.projection_replays == 4


@torch.inference_mode()
def test_native_optimized_batch_cache_and_fused_norm_match_reference():
    torch.manual_seed(42)
    model = _draft(_architecture()).to(device="npu", dtype=torch.bfloat16)
    reference = copy.deepcopy(model)
    adapter = AscendLEagleSpeculator.__new__(AscendLEagleSpeculator)
    model.configure_inference(fused_projections=True, rms_norm=adapter.create_rmsnorm_impl())
    context = LEagleContext(3, 16, 32, dtype=torch.bfloat16, device="npu")
    execution = LEagleExecution(model, context, 16, 3, LEagleExecutionOptions(), dtype=torch.bfloat16, device="npu")
    manager = LEagleOptimizedGraphRunner(
        execution, NativeLEagleGraphBackend(torch.npu, torch.device("npu:0"), torch.npu.NPUGraph)
    )
    manager.capture()
    anchors = torch.tensor([[3], [5], [7]], device="npu")
    history = {}
    for rows in ([(2, 0, 4), (0, 0, 2)], [(0, 2, 3), (2, 4, 2)]):
        inputs = [torch.randn(count, 96, device="npu", dtype=torch.bfloat16) for _, _, count in rows]
        positions = torch.cat([torch.arange(start, start + count, device="npu") for _, start, count in rows])
        execution.project(torch.cat(inputs), positions, manager)
        begin = 0
        for (slot, start, count), features in zip(rows, inputs):
            execution.update(slot, str(slot), start, begin, begin + count)
            history[slot] = features if start == 0 else torch.cat((history[slot][:start], features))
            begin += count
        result = torch.full((2, 3), -1, dtype=torch.long, device="npu")
        execution.propose([(row, slot) for row, (slot, _, _) in enumerate(rows)], anchors, result, 3, manager)
        for row, (slot, _, _) in enumerate(rows):
            prefix = reference.project_context(history[slot][None])
            expected = reference.propose_context(prefix, anchors[slot : slot + 1], 3)
            torch.testing.assert_close(result[row : row + 1], expected)
    assert manager.proposal_replays == manager.projection_replays == 2


def _count_target_replays(worker):
    manager = worker.model_runner.cudagraph_manager
    assert manager.graphs, "Target FULL graph capture was skipped"
    original = manager.run_fullgraph
    manager.leagle_test_replays = 0

    def run(*args, **kwargs):
        manager.leagle_test_replays += 1
        return original(*args, **kwargs)

    manager.run_fullgraph = run


def _graph_stats(worker):
    runner = worker.model_runner
    draft = runner.speculator.graph_runner
    return dict(
        target=runner.cudagraph_manager.leagle_test_replays,
        draft=draft.proposal_replays,
        projection=draft.projection_replays,
    )


def _serve(mode, target, draft, output_path):
    # Worker subprocesses import the serving stack after selecting V2.
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    from vllm import LLM, SamplingParams

    kwargs = dict(
        model=target,
        skip_tokenizer_init=True,
        dtype="bfloat16",
        max_model_len=128,
        max_num_seqs=2,
        max_num_batched_tokens=64,
        enforce_eager=mode == "eager",
        enable_prefix_caching=False,
        gpu_memory_utilization=0.4,
    )
    if draft != "none":
        kwargs["speculative_config"] = dict(method="leagle", model=draft, num_speculative_tokens=3)
    if mode != "eager":
        kwargs["compilation_config"] = dict(cudagraph_mode=mode, cudagraph_capture_sizes=[4, 8, 16])
    llm = LLM(**kwargs)
    if mode != "eager":
        llm.collective_rpc(_count_target_replays)
    params = SamplingParams(temperature=0, max_tokens=12, ignore_eos=True, detokenize=False)
    outputs = []
    for prompts in (([2, 3, 4], [7] * 17), ([8] * 37, [9])):
        responses = llm.generate([{"prompt_token_ids": p} for p in prompts], params)
        outputs.extend([list(response.outputs[0].token_ids) for response in responses])
    stats = llm.collective_rpc(_graph_stats) if mode != "eager" else []
    Path(output_path).write_text(json.dumps(dict(outputs=outputs, stats=stats)), encoding="utf-8")


@pytest.mark.parametrize("mode", ["FULL", "FULL_DECODE_ONLY"])
def test_target_and_draft_full_match_target_only_and_eager(tmp_path, mode):
    torch.manual_seed(42)
    config = _architecture()
    target_config = Qwen3Config(**config["transformer_layer_config"])
    target_config.num_hidden_layers = 4
    target_config.layer_types = ["full_attention"] * 4
    target = Qwen3ForCausalLM(target_config).eval()
    target_path, draft_path = tmp_path / "target", tmp_path / "draft"
    target.save_pretrained(target_path)
    draft_path.mkdir()
    draft = _draft(config)
    with torch.no_grad():
        draft.embed_tokens.weight.copy_(target.model.embed_tokens.weight)
        draft.lm_head.weight.copy_(target.lm_head.weight)
    save_file(draft.state_dict(), draft_path / "model.safetensors")
    checkpoint = dict(
        config,
        speculators_model_type="leagle",
        architectures=["LEagleDraftModel"],
        speculators_config=dict(
            algorithm="leagle",
            default_proposal_method="greedy",
            proposal_methods=[dict(speculative_tokens=3)],
            verifier=dict(name_or_path=str(target_path)),
        ),
    )
    (draft_path / "config.json").write_text(json.dumps(checkpoint), encoding="utf-8")
    results = []
    for index, (run_mode, run_draft) in enumerate(
        (
            ("eager", "none"),
            ("eager", str(draft_path)),
            (mode, str(draft_path)),
        )
    ):
        output = tmp_path / f"result_{index}.json"
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                run_mode,
                str(target_path),
                run_draft,
                str(output),
            ],
            check=True,
            timeout=300,
        )
        results.append(json.loads(output.read_text(encoding="utf-8")))
    assert results[0]["outputs"] == results[1]["outputs"] == results[2]["outputs"]
    assert results[2]["stats"]
    assert all(all(value > 0 for value in stats.values()) for stats in results[2]["stats"])


if __name__ == "__main__":
    _serve(*sys.argv[1:])
