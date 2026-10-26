# SPDX-License-Identifier: Apache-2.0
"""Compare Qwen3 Target NONE/FULL in separate processes, with optional MMuse."""

import argparse
import json
import time
from pathlib import Path


def install_replay_counter(worker):
    manager = getattr(worker.model_runner, "cudagraph_manager", None)
    if manager is None:
        raise RuntimeError("This validation requires Model Runner V2")
    counts = {"uniform_decode": 0, "mixed": 0}
    worker._target_full_validation_counts = counts
    original = manager.run_fullgraph

    def replay(desc):
        result = original(desc)
        key = "uniform_decode" if desc.uniform_token_count is not None else "mixed"
        counts[key] += 1
        return result

    manager.run_fullgraph = replay


def read_replay_counter(worker):
    manager = worker.model_runner.cudagraph_manager
    return {
        "runner": type(worker.model_runner).__name__,
        "manager": type(manager).__name__,
        "specialize_full_decode": manager.specialize_full_decode,
        "captured": [str(desc) for desc in manager.graphs],
        "replays": worker._target_full_validation_counts,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft", help="Optional MMuse checkpoint; draft stays eager")
    parser.add_argument("--graph-mode", choices=["NONE", "FULL"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    query_len = 8 if args.draft else 1
    sizes = sorted({query_len * n for n in range(1, args.max_num_seqs + 1)})
    compilation = {
        "mode": 0,
        "cudagraph_mode": args.graph_mode,
        "cudagraph_capture_sizes": sizes,
    }
    speculative = None
    if args.draft:
        speculative = {
            "method": "mmuse",
            "model": args.draft,
            "num_speculative_tokens": 7,
            "enforce_eager": True,
            "draft_sample_method": "greedy",
        }
    llm = LLM(
        model=args.model,
        enforce_eager=False,
        compilation_config=compilation,
        speculative_config=speculative,
        tensor_parallel_size=args.tp,
        max_num_seqs=args.max_num_seqs,
        max_model_len=2048,
        max_num_batched_tokens=2048,
        enable_prefix_caching=False,
        seed=0,
    )
    prompts = [
        "Explain how a computer executes a program, step by step.",
        "Describe the water cycle and explain why each stage matters.",
        "Explain how to solve a quadratic equation with an example.",
        "Compare a linked list with an array, including their costs.",
    ]
    params = SamplingParams(temperature=0, max_tokens=args.max_tokens, ignore_eos=True)
    for _ in range(2):
        llm.generate(prompts, params, use_tqdm=False)
    llm.collective_rpc(install_replay_counter)
    start = time.perf_counter()
    outputs = llm.generate(prompts, params, use_tqdm=False)
    elapsed = time.perf_counter() - start
    workers = llm.collective_rpc(read_replay_counter)
    tokens = [list(output.outputs[0].token_ids) for output in outputs]
    result = {
        "model": args.model,
        "draft": args.draft,
        "compilation": compilation,
        "seconds": elapsed,
        "output_tokens_per_second": sum(map(len, tokens)) / elapsed,
        "prompt_token_ids": [list(output.prompt_token_ids) for output in outputs],
        "output_token_ids": tokens,
        "workers": workers,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if not k.endswith("token_ids")}, indent=2))
    if args.graph_mode == "FULL" and not any(w["replays"]["uniform_decode"] for w in workers):
        raise RuntimeError("No specialized Target decode replay observed; do not report FULL validation passed")


if __name__ == "__main__":
    main()
