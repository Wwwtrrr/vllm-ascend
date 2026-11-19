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

## INT8 full-vocabulary draft head

Set `leagle_lm_head_quantization="w8a8"` in `speculative_config` to create a
private online INT8 head from the loaded BF16 draft weights. It keeps all draft
vocabulary rows, uses per-row weight scales and per-token activation scales,
then calls native `npu_dynamic_quant` and `npu_quant_matmul` for BF16 logits.
The original target/draft weights are retained; no training or checkpoint
conversion is required. Native operator references:
[dynamic quant](https://github.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu/torch_npu-npu_dynamic_quant.md)
and [quantized matmul](https://github.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu/torch_npu-npu_quant_matmul.md).

This is an Ascend LEAGLE implementation of the private low-precision proposal
head idea, not a direct port of PR #59973's FP8/NVFP4 kernels. Initial support
requires BF16, hidden/vocabulary shard sizes divisible by 32 and the LEAGLE
greedy proposal restrictions. An enabled trained candidate head quantizes its
own up-projection. Disable candidate scoring to quantize the full LM head.
Default is disabled; use `None`/omit the option for the BF16 baseline.

Weights are quantized in bounded FP32 chunks during model loading, and NZ
packing follows the existing Ascend weight-format policy. Private buffers are
nonpersistent and add memory on top of the retained BF16 weights. They are
allocated before profiling/capture. Both eager and FULL use the same operators;
hot reload requires engine restart and recapture. No unsupported-kernel CPU
fallback or automatic acceptance-preserving promise is made.

To release unused BF16 source weights after successful quantization, add
`"leagle_release_quantized_source": True` to the Python speculative config
(use lowercase `true` in JSON). Default is false, preserving existing behavior.
Full-head W8A8 releases the dense draft head. Candidate W8A8 releases the BF16
up-projection; the dense original head stays for candidate reranking, and is
released only when reranking is disabled. The BF16 trained down-projection and
target weights remain unchanged.

For vocabulary 151936, hidden 2560 and trained rank 640, this frees about
742 MiB (full head), 185 MiB (candidate reranking), or 927 MiB (direct candidate
scoring), relative to retaining both representations. Release occurs before
profiling/capture, does not reduce quantization-time peak memory, and does not
guarantee avoiding OOM. Allocator-reserved storage may remain visible but is
reusable. Restart to switch modes; no hot reload or BF16 fallback afterwards.
The figures above are for TP=1; per-rank savings with TP follow the local shard
size, approximately divided by TP with vocabulary padding overhead.
Checkpoint files are not modified. Isolated benchmarks keep their BF16 baseline.

### Trained candidate-r640 with W8A8

```python
speculative_config = {
    "method": "leagle",
    "model": "/path/to/leagle-candidate-r640",
    "num_speculative_tokens": 7,
    "leagle_use_candidate_head": True,
    "leagle_candidate_rerank": False,
    "leagle_lm_head_quantization": "w8a8",
    "leagle_release_quantized_source": True,
}
```

Rank comes from the checkpoint's `candidate_head_rank`. The trained
down-projection stays BF16, while the trained rank-to-vocab
up-projection uses the same native dynamic INT8 quantization and matmul as the
other W8A8 paths. Direct scoring takes ArgMax of the resulting BF16 logits.
For TopK/exact rescoring, use `leagle_candidate_rerank=True` and optionally set
`leagle_candidate_top_k`. Omit TopK when rerank is false.

Source release defaults to false. When enabled, it removes the BF16 trained
up-projection after successful quantization; direct scoring also releases the
unused dense draft head. Exact rescoring keeps that dense head. For vocab
151936/rank 640, the up-projection source is about 185 MiB at TP=1; the dense
head is about 742 MiB at hidden size 2560. These are freed source bytes, not
total memory or peak-memory savings. With TP, each worker quantizes/releases
its own vocabulary shard. No checkpoint conversion or retraining is required.
Runtime scoring changes and weight reload require restart/recapture.

The existing BF16/rank alignment constraints apply. Native tests cover trained
candidate W8A8 with both rerank settings, source release, TP=1/2 eager/FULL, and
width-640 INT8 graph replay. Hardware tests still require actual NPUs; evaluate
real-checkpoint candidate recall, acceptance and TPOT before deployment.

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

## Tensor-parallel LEAGLE

Target and draft use the same TP group. For two NPUs:

```python
llm = LLM(
    model="/path/to/Qwen3-4B",
    tensor_parallel_size=2,
    enable_prefix_caching=False,
    enforce_eager=False,
    compilation_config={
        "cudagraph_mode": "FULL",
        "pass_config": {"enable_sp": False},
    },
    speculative_config={
        "method": "leagle",
        "model": "/path/to/leagle/checkpoint",
        "draft_tensor_parallel_size": 2,
        "num_speculative_tokens": 7,
    },
)
```

Select both devices with `ASCEND_RT_VISIBLE_DEVICES=0,1` and enable Model Runner
V2 as in the examples above. Draft TP can be omitted to inherit target TP;
target TP=2 with draft TP=1 is rejected. Eager uses the same sharded model.

Full training checkpoints are sliced on loading; no sharded checkpoint export
or retraining is needed. Attention Q heads and MLP intermediate channels split
across ranks. KV heads split when possible, and replicate when TP exceeds the
KV head count. Embedding/LM head/candidate up-projections partition vocabulary
rows, padding each local shard to multiples of 32. Normalization, auxiliary
feature/recurrent projections, low-rank down-projections and token maps remain
replicated. Static/context/draft KV buffers use local KV heads; projected hidden
context buffers retain the full hidden dimension.

Attention/MLP row outputs use the framework TP reductions. Full/coarse logits
are gathered before selection; TopK candidate IDs are broadcast from rank zero,
and exact scores are reduced from the owning original-head shards. W8A8 quantizes each local full-head or
trained candidate up-projection shard; the opt-in source release flag continues to work. This does not gather BF16
weight matrices on each draft step.

PP/DCP/PCP must remain one, sequence parallel target features are rejected,
and target/draft TP must match. Attention head and MLP sizes must be compatible
with TP. FULL captures the sharded SDPA rollout and its collective operations.
HCCL capture and replay require validation on the installed CANN/torch_npu
stack. CPU TP=2/4 algebra and real two-process Gloo tests establish CPU
correctness, not native graph correctness or performance. Small-batch TP can
be slower due to communication; compare acceptance and TPOT before deployment.

The native serving tests below include TP=1 and TP=2 for dense, candidate,
W8A8 and trained candidate+W8A8 paths. TP=2 cases skip without two visible NPUs. Run with
`ASCEND_RT_VISIBLE_DEVICES=0,1` to validate both workers' shard dimensions,
proposal agreement and target/draft FULL replay counters.

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

Inference-time decomposition has been removed. Delete the obsolete
`leagle_candidate_svd_rank` option from existing launch configurations.

## Device-length FA3 for Qwen3 target attention

The `leagle-fa3-pr21444` branches pair the vLLM revision pinned by
[vLLM Ascend #21444](https://github.com/vllm-project/vllm-ascend/pull/21444)
with that PR's Ascend backend. Install both branches in the same environment
using the [Ascend source installation guide](../../getting_started/installation.md),
then use `flash-attn-npu>=0.5.1.post1`. This version provides the
metadata-driven dispatch needed for FULL graphs with dense paged KV.

For a Qwen3 target in FULL mode, add
`additional_config={"leagle_target_device_fa3": True}` to `LLM`, with
`block_size=128` and the attention backend left on automatic selection. This
requires an A2/A3 device, FP16 or BF16 matching the KV dtype, and the standard
V2 runner. The runner checks that every target attention group uses FA3 before
skipping the CPU length snapshot.

Target FA3 reads the rejection-corrected NPU sequence lengths and prepares
its tiling metadata before replay. Regular request scheduling and the query
offsets on the host still participate in input preparation. The draft follows
the checkpoint's existing LEAGLE path.

Run the real-checkpoint A/B on one NPU using matching FIA and FA3 runs.

```bash
ASCEND_RT_VISIBLE_DEVICES=7 \
LEAGLE_E2E_TARGET_MODEL=/path/to/Qwen3-4B \
LEAGLE_E2E_R640_MODEL=/path/to/leagle-candidate-r640 \
LEAGLE_E2E_5HS_MODEL=/path/to/leagle-5hs \
python -m pytest -s -q tests/e2e/pull_request/one_card/aclgraph/test_leagle_full.py \
  -k real_qwen3_leagle_fa3
```

The test checks target and draft graph replays and identical greedy output
tokens. It reports mean TPOT, mean ITL, and average acceptance length for each
checkpoint and backend. To capture matching traces with an existing
`LLM.start_profile()` benchmark, run it once with the option set to `False`
and once with it set to `True`. In the Host trace, compare the synchronize
associated with the speculative sequence-length copy. In the device trace,
compare the tiling and target graph replay on the same decode rounds.
