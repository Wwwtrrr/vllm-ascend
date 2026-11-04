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
        if args.try_fused:
            from leagle_lm_head_triton import FusedLMHeadPrototype, validate_edge_cases

            validate_edge_cases(device, args.fused_block_vocab, args.fused_block_k)
            print("FUSED EDGE CASES: tail masks, cross-tile ties, batch=2, NaN, hierarchical reduction passed")

            fused = FusedLMHeadPrototype(weight, args.batch_size, args.fused_block_vocab, args.fused_block_k)
            # Compare token IDs, including all-equal logits and mutated inputs.
            original_hidden = hidden.clone()
            mismatches = 0
            for case in range(args.correctness_cases + 1):
                if case == 0:
                    hidden.zero_()
                else:
                    hidden.normal_()
                expected = F.linear(hidden, weight).argmax(-1)
                actual = fused(hidden)
                mismatches += int((actual != expected).sum().item())
            hidden.copy_(original_hidden)
            report["fused_validation"] = {
                "block_vocab": args.fused_block_vocab,
                "block_k": args.fused_block_k,
                "cases": args.correctness_cases + 1,
                "token_mismatches": mismatches,
                "note": "Finite random inputs plus zero input; not a model-quality evaluation",
            }
            print("FUSED VALIDATION:", report["fused_validation"])
            if mismatches:
                raise RuntimeError("Fused token mismatch: reject this configuration; no speed claim")

            def fused_forward():
                for _ in range(args.steps):
                    output = fused(hidden)
                return output

            eager = measure(torch, fused_forward, args.warmup, args.repeats, args.trials, args.steps)
            captured = capture(torch, fused_forward, device)
            graph, output, stream, pool = captured
            # Verify replay reads changed inputs rather than capture-time values.
            hidden.normal_()
            graph.replay()
            torch.npu.synchronize()
            torch.testing.assert_close(output, F.linear(hidden, weight).argmax(-1))
            hidden.copy_(original_hidden)
            graphed = measure(torch, graph.replay, args.warmup, args.repeats, args.trials, args.steps)
            for mode, result in (("eager", eager), ("graph", graphed)):
                report["results"][f"fused_{mode}"] = result
                ms = result["device_interval_ms_per_step_median"]
                baseline = report["results"][f"linear_argmax_{mode}"]["device_interval_ms_per_step_median"]
                print(f"fused          {mode:5s}: {ms:.6f} ms/step; speedup vs linear_argmax {baseline / ms:.3f}x")
            torch.npu.synchronize()
            del output, graph, stream, pool, captured
        print("Weight bytes / interval is an estimate, not measured HBM bandwidth.")
        if args.output:
            args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
