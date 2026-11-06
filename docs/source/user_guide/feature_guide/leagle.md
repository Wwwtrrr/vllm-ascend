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

Trained candidate-head checkpoints (`candidate_head=true`) are supported through
the shared LEAGLE model in eager and native FULL ACL graphs. Low-rank retrieval
selects Top-K rows of the original LM head, followed by exact candidate scoring
and greedy selection without full-vocabulary logits. The checkpoint's rank and
Top-K are preserved, including the two trained candidate projection weights.
Dense checkpoints continue using the original dense head. Missing or non-finite
candidate weights fail loading rather than falling back to random retrieval.

For dense A/B evaluation of a candidate checkpoint, set
`leagle_use_candidate_head=false` in `speculative_config`. To adjust retrieval
width, use `leagle_candidate_top_k` with the candidate path enabled. These
settings take effect before capture. Greedy proposal verification remains the
framework's one-hot rejection path. Candidate retrieval can miss the dense best
token; measure acceptance and TPOT on the real checkpoint. Hardware tests cover
candidate eager/FULL output parity and replay counters but must be run on NPU.

Dense checkpoints also support inference-only SVD initialization. Set
`leagle_candidate_svd_rank=256` and `leagle_candidate_top_k=256` in the
speculative config. No trained candidate weights or checkpoint edits are
required: loaded LM-head weights generate CPU FP64 Gram/eigh factors, matching
the training repository's SVD method. Factor buffers are nonpersistent and
allocated on the model device before profiling and ACL capture. Replay only
executes low-rank retrieval, Top-K and original-weight candidate rescoring.

For direct low-rank logits without rescoring, set
`leagle_candidate_rerank=false` and omit `leagle_candidate_top_k`. For dense
comparison, remove `leagle_candidate_svd_rank` and set
`leagle_use_candidate_head=false`. Restart between modes; do not hot-reload
weights into an initialized SVD head. Both candidate and direct low-rank modes
can change draft tokens and acceptance. Real NPU correctness and performance
must be measured; CPU tests do not establish ACL or latency behavior.

SVD does not require NPU FP64/eigh support. CPU decomposition can take time,
with initialization start/end logs. The original dense head stays resident;
factors add device memory and are recomputed at every startup, not cached on
disk or added to saved checkpoints.

## Target input preparation graphs

`leagle_target_input_graphs` defaults to `true` for Ascend LEAGLE FULL decode.
A separate native ACL preparation graph captures request-map expansion,
position/sequence-length preparation, sampled/draft token combination, block
table gathering and slot-mapping calculation. The target model then replays
its existing FULL graph. These are separate graph executions, not one combined
end-to-end engine graph.

Host request maps, logits boundaries, FIA query boundaries, prefill lengths and
block counts are packed into one fixed-address upload. In particular, rotating
UVA prefill/block-count buffers are not captured directly. Each real request
count has a startup capture, sharing a private preparation pool. Outputs retain
the existing target input/block-table addresses, including padding. Profiling
captures are discarded and rebuilt through the speculator manager lifecycle;
serving never captures a new shape lazily.

This fast path requires FULL decode, one KV-cache group, real request slots,
one bonus token per request, no PCP and no adaptive verification. Prefill,
mixed batches, additional FIA padding rows beyond capacity and unsupported
layouts use normal preparation. Set `leagle_target_input_graphs=false` for a
same-workload A/B test. Restart the engine after changing this setting.

The operations still appear in device profiling, now under preparation graph
replay. Their occurrence counts need not decrease: this change reduces host
submissions and consolidates metadata uploads. Scheduler work, staged block
table writes, rejection/CPU length synchronization, and external input copies
remain outside the graph. Measure end-to-end TPOT with profiling disabled as
well as graph replay counters; hardware speedup is not established by CPU tests.

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

FULL adds padded batch workspaces, feature projection buffers and
graph memory. These inputs are allocated before memory profiling. Profiling
graphs are released and recaptured through the framework's graph-manager
lifecycle. Requests are drafted in batch/context buckets, each batch using one
K-token rollout replay. Verdict/position copies and request staging remain
outside graphs and still synchronize once per batch. No graph or speedup claim is established by
the CPU tests alone.

In the optimized FULL path, projection graphs write their outputs into
persistent feature/KV buffers. Separate native ACL cache-update graphs scatter
verified suffixes into request slots, including partial prefills without draft
proposals. Capture and padded writes target a private sink slot, protecting
live prefixes. The rollout graph gathers anchors by device slot indices;
slot/length metadata is staged together. Host request validation, grouping,
input copies and verdict synchronization remain outside graphs. Each nonempty
request suffix adds a cache-update replay, so measure its latency on the NPU.
The sink slot and additional graph buckets also consume device memory.

The optimized executor enables these `speculative_config` options by default:
`leagle_reuse_context_kv`, `leagle_static_draft_kv`, `leagle_batch_drafts`,
`leagle_fused_projections` and `leagle_fused_rmsnorm`. The last option dispatches
to native NPU RMSNorm. Packed Q/K/V and gate/up projections retain the original
checkpoint names and require no retraining. Set all five options to `false` for
the reference path when measuring acceptance, TPOT and profiling differences.

New verified target features are projected into persistent per-request K/V by
the projection graph; subsequent rounds reuse that prefix. Rejections, rewinds
and request-slot reuse update the valid prefix rather than promoting synthetic
draft states. The rollout graph batches requests, copies context into reusable
attention workspaces once per round and updates short logical-layer draft KV
in place. Seven draft tokens still require seven autoregressive steps.

Persistent K/V, packed weights, batched workspaces and graph buckets consume
additional memory. Start with small request/length limits and check available
memory before increasing them. Cached K/V requires fixed runtime RoPE scaling.
These changes do not add paged FIA attention. Training chunked loss and DDP
remain training-repository settings, not vLLM/Ascend serving options.

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
