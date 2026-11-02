# LEAGLE

Install the corresponding modified vLLM and vLLM Ascend checkouts together.
`method="leagle"` selects `AscendLEagleSpeculator`, which shares the upstream
LEAGLE checkpoint loader and token/state/loop recurrence and executes PyTorch
SDPA on the NPU. It does not reuse the EAGLE3 proposer.

```bash
ASCEND_RT_VISIBLE_DEVICES=0 VLLM_USE_V2_MODEL_RUNNER=1 \
vllm serve /path/to/Qwen3-4B \
  --enforce-eager --no-enable-prefix-caching \
  --tensor-parallel-size 1 --max-num-seqs 1 --max-model-len 4096 \
  --speculative-config '{"method":"leagle","model":"/path/to/leagle/checkpoints/checkpoint_best","num_speculative_tokens":7,"draft_sample_method":"greedy"}'
```

The checkpoint must be trained with `--speculator-type leagle`. Its saved config
controls decoder layers, shared loops, auxiliary target features, attention and
vocabulary mapping. A matching verifier is required. No extra architecture flags
or checkpoint conversion are needed.

This first implementation requires Model Runner V2, eager execution, a Qwen3
target and TP=PP=DCP=PCP=1. It supports greedy draft proposals with the framework's
normal target verification. Prefix caching, remote KV transfer, target LoRA,
watermarking, quantized drafts, parallel drafting and adaptive verification are
not supported. It reserves a dense projected target-feature buffer sized by
`max_num_seqs * max_model_len * hidden_size`; it rebuilds draft KV each proposal
round and does not use ACL graphs or paged FIA. Verdict/position copies cause
host/device synchronization once per batch in this eager reference path.

Run output-parity and acceptance evaluations on an NPU with the actual trained
checkpoint before measuring serving performance. The CPU factory test is:

```bash
python -m pytest --confcutdir=tests/standalone -o addopts='' \
  tests/standalone/test_leagle_dispatch.py -q
```
