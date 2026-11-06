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
from vllm_ascend.worker.v2.spec_decode.leagle_inputs import LEagleTargetInputGraphRunner, LEagleTargetInputState


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
@pytest.mark.parametrize("candidate", [False, True, "svd", "svd_lowrank"])
def test_native_optimized_batch_cache_and_fused_norm_match_reference(candidate):
    torch.manual_seed(42)
    config = _architecture()
    if candidate is True:
        config.update(candidate_head=True, candidate_head_rank=8, candidate_head_top_k=7)
    model = _draft(config).to(device="npu", dtype=torch.bfloat16)
    if candidate in ("svd", "svd_lowrank"):
        model.configure_output_head(svd_rank=8, top_k=7 if candidate == "svd" else None, rerank=candidate == "svd")
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
            execution.update(slot, str(slot), start, begin, begin + count, manager)
            history[slot] = features if start == 0 else torch.cat((history[slot][:start], features))
            begin += count
        result = torch.full((2, 3), -1, dtype=torch.long, device="npu")
        execution.propose([(row, slot) for row, (slot, _, _) in enumerate(rows)], anchors, result, 3, manager)
        for row, (slot, _, _) in enumerate(rows):
            prefix = reference.project_context(history[slot][None])
            expected = reference.propose_context(prefix, anchors[slot : slot + 1], 3)
            torch.testing.assert_close(result[row : row + 1], expected)
    assert manager.proposal_replays == manager.projection_replays == 2
    assert manager.update_replays == 4


@torch.inference_mode()
def test_native_target_input_graph_matches_eager_with_slot_reorder_and_padding():
    from vllm.v1.worker.gpu.input_batch import (
        combine_sampled_and_draft_tokens,
        expand_idx_mapping,
        prepare_pos_seq_lens,
    )

    buffers = SimpleNamespace(
        input_ids=torch.full((16,), -1, device="npu", dtype=torch.long),
        positions=torch.zeros(16, device="npu", dtype=torch.long),
        seq_lens=torch.zeros(3, device="npu", dtype=torch.int32),
        query_start_loc=torch.zeros(5, device="npu", dtype=torch.int32),
    )
    requests = SimpleNamespace(
        num_computed_tokens=SimpleNamespace(gpu=torch.tensor([8, 12, 5], device="npu", dtype=torch.int32)),
        prefill_len=SimpleNamespace(gpu=torch.tensor([2, 3, 1], device="npu", dtype=torch.int32), np=[2, 3, 1]),
        last_sampled_tokens=torch.tensor([5, 7, 9], device="npu"),
        draft_tokens=torch.arange(6, device="npu").view(3, 2),
    )
    state = LEagleTargetInputState(buffers, requests, 3, 3)
    from vllm_ascend.worker.v2.block_table import AscendBlockTables

    blocks = AscendBlockTables([4], 3, 16, [8], torch.device("npu:0"), kernel_block_sizes=[4])
    blocks.block_tables[0].gpu.copy_(torch.arange(24, device="npu", dtype=torch.int32).view(3, 8))
    blocks.num_blocks.np.fill(8)
    blocks.num_blocks.copy_to_uva()
    state.block_tables = blocks
    manager = LEagleTargetInputGraphRunner(
        state, NativeLEagleGraphBackend(torch.npu, torch.device("npu:0"), torch.npu.NPUGraph)
    )
    manager.capture()
    for slots, cu, query in (
        ([2, 0], [0, 3, 4], [0, 3, 4, 8, 4]),
        ([0, 2], [0, 2, 5], [0, 2, 5, 8, 5]),
        ([1], [0, 1], [0, 1, 1, 1, 1]),
    ):
        # Advance device-side state between replays, including the one-token
        # no-draft case and changing request counts without new captures.
        requests.num_computed_tokens.gpu.add_(1)
        blocks.num_blocks.np[0, 0] = 7
        blocks.num_blocks.copy_to_uva()
        state.stage(slots, cu, query)
        n, total = len(slots), cu[-1]
        buffers.input_ids.fill_(-1)
        buffers.query_start_loc.copy_(state.query)
        expected = expand_idx_mapping(state.idx[:n], total, state.cu[: n + 1], 3)
        prepare_pos_seq_lens(
            state.idx[:n],
            buffers.query_start_loc,
            requests.num_computed_tokens.gpu,
            buffers.positions,
            buffers.seq_lens,
        )
        logits = combine_sampled_and_draft_tokens(
            buffers.input_ids,
            state.idx[:n],
            requests.last_sampled_tokens,
            buffers.query_start_loc,
            buffers.seq_lens,
            requests.prefill_len.gpu,
            requests.draft_tokens,
            state.cu[: n + 1],
            total,
            1,
        )
        expected = tuple(t.clone() for t in (*expected, logits))
        expected_tables = tuple(t.clone() for t in blocks.gather_block_tables(state.idx[:n], 3))
        expected_mappings = blocks.compute_slot_mappings(
            state.idx[:n], buffers.query_start_loc, buffers.positions, 16
        ).clone()
        expected_ids, expected_pos, expected_seq = (
            buffers.input_ids[:total].clone(),
            buffers.positions[:total].clone(),
            buffers.seq_lens.clone(),
        )
        buffers.input_ids.fill_(-1)
        actual = manager.replay(n, total)
        for value, reference in zip(actual, expected):
            torch.testing.assert_close(value, reference)
        torch.testing.assert_close(buffers.input_ids[:total], expected_ids)
        torch.testing.assert_close(buffers.positions[:total], expected_pos)
        torch.testing.assert_close(buffers.seq_lens, expected_seq)
        actual_tables, actual_mappings = manager.prepared_attn
        for value, reference in zip(actual_tables, expected_tables):
            torch.testing.assert_close(value, reference)
        torch.testing.assert_close(actual_mappings, expected_mappings)
    assert manager.replays == 3


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
        cache_update=draft.update_replays,
        input_prep=getattr(getattr(runner.speculator, "target_input_graph_runner", None), "replays", 0),
    )


def _serve(mode, target, draft, output_path, svd="none", input_graphs="on"):
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
        kwargs["speculative_config"]["leagle_target_input_graphs"] = input_graphs == "on"
        if svd != "none":
            kwargs["speculative_config"].update(leagle_candidate_svd_rank=8, leagle_candidate_rerank=svd == "svd")
            if svd == "svd":
                kwargs["speculative_config"]["leagle_candidate_top_k"] = 7
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
@pytest.mark.parametrize("candidate", [False, True, "svd", "svd_lowrank"])
def test_target_and_draft_full_match_target_only_and_eager(tmp_path, mode, candidate):
    torch.manual_seed(42)
    config = _architecture()
    if candidate is True:
        config.update(candidate_head=True, candidate_head_rank=8, candidate_head_top_k=7)
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
    for index, (run_mode, run_draft, input_graphs) in enumerate(
        (
            ("eager", "none", "on"),
            ("eager", str(draft_path), "on"),
            (mode, str(draft_path), "on"),
            (mode, str(draft_path), "off"),
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
                candidate if candidate in ("svd", "svd_lowrank") else "none",
                input_graphs,
            ],
            check=True,
            env={**os.environ, "VLLM_ALLOW_INSECURE_SERIALIZATION": "1"},
            timeout=300,
        )
        results.append(json.loads(output.read_text(encoding="utf-8")))
    assert results[0]["outputs"] == results[1]["outputs"] == results[2]["outputs"] == results[3]["outputs"]
    assert results[2]["stats"]
    assert all(all(value > 0 for value in stats.values()) for stats in results[2]["stats"])
    assert all(stats["input_prep"] == 0 for stats in results[3]["stats"])


if __name__ == "__main__":
    _serve(*sys.argv[1:])
