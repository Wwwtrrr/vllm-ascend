# Qwen3 Target FULL graph validation

This change specializes uniform Qwen3 decode batches in Model Runner V2's
FULL mode. It does not enable Draft graph execution, change sampling, or alter
Attention task update ordering. Both repositories must contain the matching
change: vLLM supplies the specialization hook and vLLM Ascend enables it for
Qwen3 fixed-length FULL execution.

The manager retains mixed FULL graphs for prefill. For the same captured token
count it prefers a uniform decode descriptor when the runtime query length
matches. Capture input preparation then receives that query length, rather than
the mixed graph's maximum query bound. This allows the backend to capture an
appropriate decode Attention path. It is not a guarantee of acceleration.
Additional graph variants can increase capture time and device memory use.

## Target-only comparison

Run each command in a fresh process in the existing Ascend environment. Select
Model Runner V2 using the environment's existing configuration. The validation
fails explicitly when no V2 graph manager is available.

```bash
python examples/qwen3_target_full_validation.py \
  --model /path/to/Qwen3-4B --graph-mode NONE --output none.json
python examples/qwen3_target_full_validation.py \
  --model /path/to/Qwen3-4B --graph-mode FULL --output full.json
```

Both cases use compilation mode 0, deterministic greedy sampling, disabled
prefix caching and the same two warmup runs. Only the subsequent generation is
timed. FULL must observe at least one uniform Target decode replay. Worker
reports include captured descriptors and uniform/mixed replay counts. The script
does not install counters until after warmup and does not change production
logging or require a diagnostic configuration option.

Compare `prompt_token_ids`, `output_token_ids`, `seconds`, and
`output_tokens_per_second` in the two JSON files. A Token mismatch needs further
investigation; equal Token IDs on this small sample alone do not establish full
correctness. Repeat runs and alternate their order before drawing performance
conclusions. Counters add a small equal Python wrapper to both cases; final
performance measurements should use the ordinary benchmark without instrumentation.

## MMuse integration comparison

After the Target-only check, repeat both commands with
`--draft /path/to/qwen_3_4b_mmuse`. The script keeps Draft eager and uses seven
speculative tokens. It derives capture sizes from eight Query tokens per request;
without a Draft it uses one. Use `--max-num-seqs 4` to test request padding and
batch transitions. Keep TP, model weights and other arguments identical between
the two cases. The current Qwen3 MMuse adapter requires TP=1.

This script validates the initial deployment path; longer, heterogeneous prompts,
request slot reuse, output correctness, acceptance statistics and a device trace
remain necessary before declaring the integration validated on NPU hardware.
