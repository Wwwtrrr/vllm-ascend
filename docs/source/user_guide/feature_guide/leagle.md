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

## INT8 full-vocabulary draft head

Set `leagle_lm_head_quantization="w8a8"` in `speculative_config` to create a
private online INT8 head from the loaded BF16 draft weights. It keeps all draft
vocabulary rows, uses per-row weight scales and per-token activation scales,
then calls native `npu_dynamic_quant` and `npu_quant_matmul` for BF16 logits.
After installation, the original BF16 draft LM head is released; the target
weights and saved checkpoint remain unchanged. No training or checkpoint
conversion is required. Native operator references:
[dynamic quant](https://github.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu/torch_npu-npu_dynamic_quant.md)
and [quantized matmul](https://github.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu/torch_npu-npu_quant_matmul.md).

This is an Ascend LEAGLE implementation of the private low-precision proposal
head idea, not a direct port of PR #59973's FP8/NVFP4 kernels. Initial support
requires BF16, hidden/vocabulary multiples of 32 and LEAGLE draft TP=1,
greedy proposal restrictions. Candidate/SVD scoring is rejected: remove SVD
rank and set `leagle_use_candidate_head=false` when using a candidate checkpoint.
Default is disabled; use `None`/omit the option for the BF16 baseline.

Weights are quantized in bounded FP32 chunks during model loading, and NZ
packing follows the existing Ascend weight-format policy. Private buffers are
nonpersistent and replace the BF16 draft head before profiling/capture. For
151936 x 2560 weights, freeing the dense head saves about 742 MiB relative to
retaining both, or about 371 MiB relative to BF16 alone, excluding scales and
packing overhead. Initialization peak memory still includes both heads, and
the allocator may reserve freed storage for reuse. Dense fallback is unavailable
until restart with W8A8 disabled. The isolated benchmark retains its BF16
baseline for comparisons; its total memory is not representative of serving.
Both eager and FULL use the same operators;
hot reload requires engine restart and recapture. No unsupported-kernel CPU
fallback or automatic acceptance-preserving promise is made.

Run the isolated benchmark before serving:

```bash
ASCEND_RT_VISIBLE_DEVICES=7 python benchmarks/leagle/benchmark_leagle_lm_head.py \
  --steps 7 --try-int8 --int8-format nz \
  --warmup 5 --repeats 20 --trials 5 --output lm_head_int8.json
```

The benchmark uses the same head implementation, with explicit NZ format;
`--int8-format nd` provides a layout comparison. It reports random-input top-1
agreement/error separately from eager/FULL timings, not model acceptance or
quality. Quantization can change draft token choices. Compare real-checkpoint
acceptance, output quality and TPOT against BF16. Native hardware tests are
provided, but cannot be replaced by CPU emulation tests.

This implementation requires Model Runner V2, a Qwen3 target and PP=DCP=PCP=1.
Target TP can exceed one: add `--tensor-parallel-size 2` for two NPUs.
The LEAGLE draft TP defaults to one. Every TP worker loads the full draft,
including context/KV buffers and any private W8A8 or candidate/SVD head.
This replicates draft memory on every NPU; it does not shard the draft LM head
or reduce its per-token projection cost. Rank-zero draft proposals are broadcast
after each real rollout, outside ACL graphs. Sequence parallel target features
are rejected. Both repositories must be updated together.

The existing accuracy test includes TP=2 cases (requires two visible NPUs):

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 pytest -sv \
  tests/e2e/pull_request/one_card/aclgraph/test_leagle_full.py \
  -k target_and_draft_full_match_target_only_and_eager
```

The TP=2 cases check target-only/eager/FULL greedy output agreement and graph
replays on both workers, including candidate, SVD and W8A8 heads. These require
actual NPU execution; CPU contract tests do not establish hardware correctness.

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
persistent feature/KV buffers. A separate native ACL cache-update graph scatters
all verified suffixes in a batch into request slots in one replay, including
partial prefills without draft proposals. Empty updates skip replay. CPU
validation packs source/destination indices once and rejects duplicate slots,
gaps and overflow before device writes. Capture and padding target private sink rows, protecting
live prefixes. The rollout graph gathers anchors by device slot indices;
slot/length metadata is staged together. Host request validation, grouping,
input copies and verdict synchronization remain outside graphs. Sixteen valid
requests now require one cache-update replay rather than sixteen. K/V storage
layout is unchanged, so generic IndexPut transpose costs are not eliminated.
Measure native NPU replay and TPOT before claiming a speedup. Sink storage grows
to `ceil(max_num_batched_tokens / max_model_len)` rows, and update buckets cover
the batch token capacity; account for that extra memory when sizing the engine.

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
