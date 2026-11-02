# LEAGLE

Install the corresponding modified vLLM and vLLM Ascend checkouts together.
`method="leagle"` selects `AscendLEagleSpeculator`, which shares the upstream
LEAGLE checkpoint loader and token/state/loop recurrence and executes PyTorch
SDPA on the NPU. It does not reuse the EAGLE3 proposer.

```bash
ASCEND_RT_VISIBLE_DEVICES=0 VLLM_USE_V2_MODEL_RUNNER=1 \
vllm serve /path/to/Qwen3-4B \
  --no-enable-prefix-caching \
  --compilation-config '{"cudagraph_mode":"FULL"}' \
  --tensor-parallel-size 1 --max-num-seqs 1 --max-model-len 4096 \
  --speculative-config '{"method":"leagle","model":"/path/to/leagle/checkpoints/checkpoint_best","num_speculative_tokens":7,"draft_sample_method":"greedy"}'
```

The checkpoint must be trained with `--speculator-type leagle`. Its saved config
controls decoder layers, shared loops, auxiliary target features, attention and
vocabulary mapping. A matching verifier is required. No extra architecture flags
or checkpoint conversion are needed.

This implementation requires Model Runner V2, a Qwen3 target and TP=PP=DCP=PCP=1.
It supports eager execution and FULL target/draft ACL graphs. Use
`--enforce-eager` without the compilation option above for eager execution.
`FULL_DECODE_ONLY` captures target verification while leaving prefill eager.
It supports greedy draft proposals with the framework's
normal target verification. Prefix caching, remote KV transfer, target LoRA,
watermarking, quantized drafts, parallel drafting and adaptive verification are
not supported. It reserves a dense projected target-feature buffer sized by
`max_num_seqs * max_model_len * hidden_size`; it rebuilds draft KV each proposal
round. Target FULL reuses the existing ModelAclGraphManager, including its
auxiliary-state outputs and attention-parameter updates. Draft FULL uses a
separate native NPUGraph pool and capture stream with fixed buffers. It captures
feature projection plus the complete K-token shared-loop rollout in shape
buckets. Each replay reads the current device context length; stale rejected
features and padding are masked, and draft KV starts fresh. Runtime-resizing
RoPE variants are rejected for draft FULL. There is no paged FIA integration.

FULL adds a padded single-request context buffer, feature projection buffers and
graph memory. These inputs are allocated before memory profiling. Profiling
graphs are released and recaptured through the framework's graph-manager
lifecycle. Requests are drafted sequentially, each using one K-token rollout
replay. Verdict/position copies and request staging remain outside graphs and
still synchronize once per batch. No graph or speedup claim is established by
the CPU tests alone.

Run output-parity and acceptance evaluations on an NPU with the actual trained
checkpoint before measuring serving performance. The CPU factory test is:

```bash
python -m pytest --confcutdir=tests/standalone -o addopts='' \
  tests/standalone/test_leagle_dispatch.py -q
```

Real NPU tests use a small local Qwen3 target/drafter (no model download):

```bash
VLLM_USE_V2_MODEL_RUNNER=1 pytest -s \
  tests/e2e/pull_request/one_card/aclgraph/test_leagle_full.py
```

They exercise native ACL replay with shrinking contexts, compare target-only,
LEAGLE eager, FULL and FULL_DECODE_ONLY outputs, and assert that both target FULL
and draft projection/rollout replay counters advance. Run a further acceptance
and latency evaluation with the actual trained checkpoint before deployment.
