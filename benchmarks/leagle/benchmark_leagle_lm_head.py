"""Standalone Ascend LM-head benchmark; no vLLM or checkpoint is required.

Example:
    ASCEND_RT_VISIBLE_DEVICES=7 python benchmark_leagle_lm_head.py
    ASCEND_RT_VISIBLE_DEVICES=7 python benchmark_leagle_lm_head.py --steps 7

Random contiguous BF16 weights match nn.Linear's [vocab, hidden] layout.
Repeated steps use the same hidden state: this isolates LM head, not a complete
autoregressive LEAGLE rollout. No device format conversion is forced.
"""

import argparse
import json
import statistics
import time
from pathlib import Path


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def measure(torch, forward, warmup, repeats, trials, steps):
    for _ in range(warmup):
        forward()
    torch.npu.synchronize()
    device_ms, wall_ms = [], []
    for _ in range(trials):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        torch.npu.synchronize()
        before = time.perf_counter()
        start.record()
        for _ in range(repeats):
            forward()
        end.record()
        torch.npu.synchronize()
        wall_ms.append((time.perf_counter() - before) * 1000 / repeats)
        device_ms.append(start.elapsed_time(end) / repeats)
    median = statistics.median(device_ms)
    return {
        "device_interval_ms_per_call_median": median,
        "device_interval_ms_per_call_min": min(device_ms),
        "device_interval_ms_per_call_max": max(device_ms),
        "device_interval_ms_per_step_median": median / steps,
        "wall_ms_per_call_median": statistics.median(wall_ms),
    }


def capture(torch, forward, device):
    current = torch.npu.current_stream(device)
    stream = torch.npu.Stream(device=device)
    pool = torch.npu.graph_pool_handle()
    stream.wait_stream(current)
    with torch.npu.stream(stream):
        for _ in range(3):
            forward()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, pool=pool, stream=stream):
            output = forward()
    current.wait_stream(stream)
    torch.npu.synchronize()
    # Retain the stream and private pool along with the graph and its outputs.
    return graph, output, stream, pool


def benchmark_fused(
    torch,
    hidden,
    weight,
    args,
    block_vocab,
    block_k,
    *,
    backend="vector",
    projection_only=False,
):
    import torch.nn.functional as F
    from leagle_lm_head_triton import FusedLMHeadPrototype, validate_edge_cases

    original_hidden = hidden.clone()
    try:
        # Every configuration sees the same validation inputs.
        torch.manual_seed(args.seed + 1)
        validate_edge_cases(
            hidden.device,
            block_vocab,
            block_k,
            backend=backend,
            projection_only=projection_only,
        )
        fused = FusedLMHeadPrototype(
            weight,
            args.batch_size,
            block_vocab,
            block_k,
            backend=backend,
            projection_only=projection_only,
        )
        mismatches = 0
        for case in range(args.correctness_cases + 1):
            if case == 0:
                hidden.zero_()
            else:
                hidden.normal_()
            expected_logits = F.linear(hidden, weight)
            actual = fused(hidden)
            if projection_only:
                torch.testing.assert_close(actual, expected_logits, rtol=0.02, atol=0.02)
                actual = actual.argmax(-1)
            mismatches += int((actual != expected_logits.argmax(-1)).sum().item())
        if mismatches:
            raise ValueError(f"Token mismatches: {mismatches}; configuration rejected")
        hidden.copy_(original_hidden)

        def forward():
            for _ in range(args.steps):
                output = fused(hidden)
            return output

        eager = measure(torch, forward, args.warmup, args.repeats, args.trials, args.steps)
        captured = capture(torch, forward, hidden.device)
        graph, output, _, _ = captured
        # Changed-input replay catches accidental capture-time specialization.
        hidden.normal_()
        graph.replay()
        torch.npu.synchronize()
        expected_logits = F.linear(hidden, weight)
        if projection_only:
            torch.testing.assert_close(output, expected_logits, rtol=0.02, atol=0.02)
            torch.testing.assert_close(output.argmax(-1), expected_logits.argmax(-1))
        else:
            torch.testing.assert_close(output, expected_logits.argmax(-1))
        hidden.copy_(original_hidden)
        graphed = measure(torch, graph.replay, args.warmup, args.repeats, args.trials, args.steps)
        torch.npu.synchronize()
        return {
            "status": "passed",
            "backend": backend,
            "projection_only": projection_only,
            "block_vocab": block_vocab,
            "block_k": block_k,
            "token_mismatches": 0,
            "validation_cases": args.correctness_cases + 1,
            "edge_cases_and_changed_input_replay": "passed",
            "eager": eager,
            "graph": graphed,
        }
    finally:
        hidden.copy_(original_hidden)
        # A device execution failure must stop the sweep, not be treated as a
        # recoverable compiler error in a potentially poisoned device context.
        torch.npu.synchronize()


def save_report(report, output):
    if output:
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        default="npu:0",
        help="Logical device after ASCEND_RT_VISIBLE_DEVICES filtering",
    )
    parser.add_argument("--batch-size", type=positive_int, default=1)
    parser.add_argument("--hidden-size", type=positive_int, default=2560)
    parser.add_argument("--vocab-size", type=positive_int, default=151936)
    parser.add_argument("--steps", type=positive_int, default=1)
    parser.add_argument("--warmup", type=positive_int, default=20)
    parser.add_argument("--repeats", type=positive_int, default=100)
    parser.add_argument("--trials", type=positive_int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--try-dot", action="store_true", help="Test BF16 tl.dot prototype")
    parser.add_argument(
        "--sweep-dot",
        action="store_true",
        help="Sweep dot tiles: vocab 64/128, K 64/128",
    )
    parser.add_argument(
        "--dot-projection-only",
        action="store_true",
        help="Compare dot logits with native Linear, without argmax",
    )
    parser.add_argument("--dot-block-vocab", type=positive_int, default=64)
    parser.add_argument("--dot-block-k", type=positive_int, default=128)
    parser.add_argument(
        "--sweep-fused",
        action="store_true",
        help="Sweep vocab tiles 8/16/32 and hidden tiles 128/256",
    )
    parser.add_argument(
        "--try-fused",
        action="store_true",
        help="Benchmark experimental Ascend Triton projection + argmax",
    )
    parser.add_argument("--fused-block-vocab", type=positive_int, default=16)
    parser.add_argument("--fused-block-k", type=positive_int, default=256)
    parser.add_argument("--correctness-cases", type=positive_int, default=8)
    parser.add_argument("--output", type=Path, help="Optional JSON results file")
    args = parser.parse_args()
    if (args.try_dot or args.sweep_dot) and (args.try_fused or args.sweep_fused):
        parser.error("Choose either dot or vector experiments in one run")
    if args.dot_projection_only and not (args.try_dot or args.sweep_dot):
        parser.error("--dot-projection-only requires --try-dot or --sweep-dot")

    import torch
    import torch.nn.functional as F

    try:
        import torch_npu
    except ImportError:
        parser.error("torch_npu is required; run this script in your NPU environment")
    if not torch.npu.is_available():
        parser.error("No NPU is available")
    torch.npu.set_device(args.device)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    with torch.inference_mode():
        hidden = torch.randn(args.batch_size, 1, args.hidden_size, device=device, dtype=torch.bfloat16)
        weight = torch.randn(args.vocab_size, args.hidden_size, device=device, dtype=torch.bfloat16)
        weight_bytes = weight.numel() * weight.element_size()
        report = {
            "torch_version": str(torch.__version__),
            "torch_npu_version": str(getattr(torch_npu, "__version__", "unknown")),
            "device": str(device),
            "weight_shape": list(weight.shape),
            "weight_stride": list(weight.stride()),
            "input_shape": list(hidden.shape),
            "dtype": str(weight.dtype),
            "weight_MiB": weight_bytes / 1024**2,
            "steps_per_call": args.steps,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "trials": args.trials,
            "results": {},
        }
        print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2))
        print("Device intervals include gaps between launches, not just kernel time.")
        print("Each trial measures repeated calls; min/median/max are trial averages.")
        print("Seven steps repeat the same input, not a full draft rollout.")

        for with_argmax in (False, True):
            label = "linear_argmax" if with_argmax else "linear"

            def forward(with_argmax=with_argmax):
                for _ in range(args.steps):
                    logits = F.linear(hidden, weight)
                    output = logits.argmax(-1) if with_argmax else logits
                return output

            reference = forward().clone()
            eager = measure(torch, forward, args.warmup, args.repeats, args.trials, args.steps)
            captured = capture(torch, forward, device)
            graph, output, stream, pool = captured
            graph.replay()
            torch.npu.synchronize()
            torch.testing.assert_close(output, reference)
            graphed = measure(torch, graph.replay, args.warmup, args.repeats, args.trials, args.steps)
            for mode, result in (("eager", eager), ("graph", graphed)):
                ms = result["device_interval_ms_per_step_median"]
                result["weight_bytes_over_interval_TB_s"] = weight_bytes / (ms / 1000) / 1e12
                report["results"][f"{label}_{mode}"] = result
                print(
                    f"{label:14s} {mode:5s}: {ms:.6f} ms/step; "
                    f"{result['device_interval_ms_per_call_median']:.6f} ms/call; "
                    f"wall {result['wall_ms_per_call_median']:.6f} ms/call"
                )
            torch.npu.synchronize()
            del output, graph, stream, pool, captured, reference
        if args.try_fused or args.sweep_fused or args.try_dot or args.sweep_dot:
            backend = "dot" if args.try_dot or args.sweep_dot else "vector"
            projection_only = args.dot_projection_only
            baseline_label = "linear" if projection_only else "linear_argmax"
            configurations = (
                [(v, k) for v in (8, 16, 32) for k in (128, 256)]
                if args.sweep_fused
                else [(args.fused_block_vocab, args.fused_block_k)]
            )
            if backend == "dot":
                configurations = (
                    [(v, k) for v in (64, 128) for k in (64, 128)]
                    if args.sweep_dot
                    else [(args.dot_block_vocab, args.dot_block_k)]
                )
            report["prototype_backend"] = backend
            report["projection_only"] = projection_only
            report["fused_configs"] = []
            save_report(report, args.output)
            baseline = report["results"][f"{baseline_label}_graph"]["device_interval_ms_per_step_median"]
            for block_vocab, block_k in configurations:
                print(
                    f"TRY {backend.upper()} projection_only={projection_only} vocab={block_vocab} k={block_k}",
                    flush=True,
                )
                try:
                    result = benchmark_fused(
                        torch,
                        hidden,
                        weight,
                        args,
                        block_vocab,
                        block_k,
                        backend=backend,
                        projection_only=projection_only,
                    )
                except Exception as error:  # noqa: BLE001 -- record failed sweep cases
                    # Re-raise asynchronous device failures instead of trying
                    # subsequent kernels on an unhealthy context.
                    torch.npu.synchronize()
                    result = {
                        "status": "failed",
                        "block_vocab": block_vocab,
                        "block_k": block_k,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                    print(f"FAILED vocab={block_vocab} k={block_k}: {error}", flush=True)
                else:
                    for mode in ("eager", "graph"):
                        ms = result[mode]["device_interval_ms_per_step_median"]
                        native = report["results"][f"{baseline_label}_{mode}"]["device_interval_ms_per_step_median"]
                        result[mode]["speedup_vs_native"] = native / ms
                        print(
                            f"fused v={block_vocab} k={block_k} {mode}: {ms:.6f} ms/step; speedup {native / ms:.3f}x",
                            flush=True,
                        )
                report["fused_configs"].append(result)
                save_report(report, args.output)
            ranked = sorted(
                (r for r in report["fused_configs"] if r["status"] == "passed"),
                key=lambda r: r["graph"]["device_interval_ms_per_step_median"],
            )
            print(f"GRAPH RANKING (native {baseline_label}: {baseline:.6f} ms/step)")
            report["fused_graph_ranking"] = []
            for rank, result in enumerate(ranked, 1):
                ms = result["graph"]["device_interval_ms_per_step_median"]
                row = {
                    "rank": rank,
                    "block_vocab": result["block_vocab"],
                    "block_k": result["block_k"],
                    "ms_per_step": ms,
                    "speedup": baseline / ms,
                }
                report["fused_graph_ranking"].append(row)
                print(f"{rank}. vocab={row['block_vocab']} k={row['block_k']}: {ms:.6f} ms/step, {baseline / ms:.3f}x")
            if not ranked:
                print("No configuration passed; no fused performance result is valid.")
            elif ranked[0]["graph"]["device_interval_ms_per_step_median"] >= baseline:
                print("No fused configuration beats the native baseline; keep the native path.")
        print("Weight bytes / interval is an estimate, not measured HBM bandwidth.")
        if args.output:
            args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
