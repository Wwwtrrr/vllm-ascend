"""Experimental Ascend Triton GEMV + greedy selection, without full logits.

This is a vector-reduction prototype, not a tuned Cube matmul replacement.
FP32 reduction order can differ from native matmul. Benchmark and validate
before considering serving integration. Requires the Ascend Triton backend.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _project_local_max(
    X,
    W,
    Scores,
    Ids,
    H: tl.constexpr,
    V: tl.constexpr,
    PARTS: tl.constexpr,
    BV: tl.constexpr,
    BK: tl.constexpr,
):
    part = tl.program_id(0)
    row = tl.program_id(1)
    vocab = part * BV + tl.arange(0, BV)
    offsets = tl.arange(0, BK)
    # Accumulate chunks in FP32, reducing the hidden dimension on chip.
    accum = tl.full((BV, BK), 0, tl.float32)
    for base in range(tl.cdiv(H, BK)):
        cols = base * BK + offsets
        x = tl.load(X + row * H + cols, cols < H, other=0).to(tl.float32)
        w = tl.load(
            W + vocab[:, None] * H + cols[None, :],
            (vocab[:, None] < V) & (cols[None, :] < H),
            other=0,
        ).to(tl.float32)
        accum = accum + w * x[None, :]
    # Reference F.linear materializes BF16 logits before argmax.
    scores = tl.sum(accum, axis=1).to(tl.bfloat16).to(tl.float32)
    nan = (scores != scores) & (vocab < V)  # noqa: PLR0124 -- IEEE NaN check
    has_nan = tl.sum(nan.to(tl.int32), axis=0) > 0
    maximum = tl.max(tl.where((vocab < V) & ~nan, scores, -float("inf")), axis=0)
    winner = tl.where(has_nan, nan, (vocab < V) & (scores == maximum))
    index = tl.min(tl.where(winner, vocab, 2147483647), axis=0)
    tl.store(Scores + row * PARTS + part, tl.where(has_nan, float("nan"), maximum))
    tl.store(Ids + row * PARTS + part, index)


@triton.jit
def _reduce_max(
    Scores,
    Ids,
    OutScores,
    OutIds,
    PARTS: tl.constexpr,
    OUT_PARTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    part = tl.program_id(0)
    row = tl.program_id(1)
    offsets = part * BLOCK + tl.arange(0, BLOCK)
    scores = tl.load(Scores + row * PARTS + offsets, offsets < PARTS, other=-float("inf"))
    ids = tl.load(Ids + row * PARTS + offsets, offsets < PARTS, other=2147483647)
    nan = (scores != scores) & (offsets < PARTS)  # noqa: PLR0124 -- IEEE NaN check
    has_nan = tl.sum(nan.to(tl.int32), axis=0) > 0
    maximum = tl.max(tl.where(nan, -float("inf"), scores), axis=0)
    winner = tl.where(has_nan, nan, (offsets < PARTS) & (scores == maximum))
    index = tl.min(tl.where(winner, ids, 2147483647), axis=0)
    tl.store(OutScores + row * OUT_PARTS + part, tl.where(has_nan, float("nan"), maximum))
    tl.store(OutIds + row * OUT_PARTS + part, index)


class FusedLMHeadPrototype:
    """Preallocated buffers permit capture after eager compilation/warmup."""

    def __init__(self, weight, batch_size=1, block_vocab=16, block_k=256):
        if weight.device.type != "npu" or weight.dtype != torch.bfloat16:
            raise ValueError("Prototype requires NPU BF16 weights")
        if weight.ndim != 2 or not weight.is_contiguous():
            raise ValueError("Expected contiguous [vocab, hidden] weights")
        if any(value < 1 or value & (value - 1) for value in (block_vocab, block_k)):
            raise ValueError("Tile dimensions must be positive powers of two")
        self.weight = weight
        self.batch = batch_size
        self.vocab, self.hidden = weight.shape
        self.bv, self.bk = block_vocab, block_k
        self.parts = triton.cdiv(self.vocab, block_vocab)
        self.scores = torch.empty(batch_size, self.parts, device=weight.device, dtype=torch.float32)
        self.ids = torch.empty(batch_size, self.parts, device=weight.device, dtype=torch.int32)
        self.output = torch.empty(batch_size, 1, device=weight.device, dtype=torch.long)
        # Bound each reduction's live vectors independently of vocabulary size.
        # 9496 candidates become 19, then 1, rather than a 16384-wide UB tile.
        self.reduce_block = 512
        self.reductions = []
        scores, ids, parts = self.scores, self.ids, self.parts
        while True:
            out_parts = triton.cdiv(parts, self.reduce_block)
            out_scores = torch.empty(batch_size, out_parts, device=weight.device, dtype=torch.float32)
            out_ids = (
                self.output
                if out_parts == 1
                else torch.empty(batch_size, out_parts, device=weight.device, dtype=torch.int32)
            )
            self.reductions.append((scores, ids, out_scores, out_ids, parts, out_parts))
            if out_parts == 1:
                break
            scores, ids, parts = out_scores, out_ids, out_parts

    def __call__(self, hidden):
        if (
            hidden.shape != (self.batch, 1, self.hidden)
            or hidden.dtype != self.weight.dtype
            or hidden.device != self.weight.device
            or not hidden.is_contiguous()
        ):
            raise ValueError("Expected contiguous BF16 [batch, 1, hidden] input")
        _project_local_max[(self.parts, self.batch)](
            hidden,
            self.weight,
            self.scores,
            self.ids,
            self.hidden,
            self.vocab,
            self.parts,
            self.bv,
            self.bk,
            enable_fp_fusion=False,
        )
        for scores, ids, out_scores, out_ids, parts, out_parts in self.reductions:
            _reduce_max[(out_parts, self.batch)](
                scores,
                ids,
                out_scores,
                out_ids,
                parts,
                out_parts,
                min(self.reduce_block, triton.next_power_of_2(parts)),
            )
        return self.output


@torch.inference_mode()
def validate_edge_cases(device, block_vocab, block_k):
    """Exercise tail masks, cross-tile ties, batches and NaN selection on NPU."""
    vocab, width = max(block_vocab, 2) + 5, block_k + 3
    weight = torch.zeros(vocab, width, device=device, dtype=torch.bfloat16)
    hidden = torch.zeros(2, 1, width, device=device, dtype=torch.bfloat16)
    boundary = max(block_vocab, 2)
    weight[boundary - 1, 0] = 2
    weight[boundary, 0] = 2
    weight[-1, 1] = 3
    hidden[0, 0, 0] = 1
    hidden[1, 0, 1] = 1
    kernel = FusedLMHeadPrototype(weight, 2, block_vocab, block_k)
    for case in range(3):
        if case == 1:
            hidden.zero_()
        elif case == 2:
            weight[-1, 0] = float("nan")
        expected = torch.nn.functional.linear(hidden, weight).argmax(-1)
        torch.testing.assert_close(kernel(hidden), expected)

    # More than 512 local candidates forces multiple reduction levels.
    weight = torch.zeros(block_vocab * 513 + 3, 17, device=device, dtype=torch.bfloat16)
    hidden = torch.ones(1, 1, 17, device=device, dtype=torch.bfloat16)
    weight[-1, 0] = 3
    kernel = FusedLMHeadPrototype(weight, 1, block_vocab, block_k)
    for case in range(3):
        if case == 1:
            weight[0, 0] = 3  # Global tie across reduction groups: choose ID zero.
        elif case == 2:
            weight[-1, 1] = float("nan")
        expected = torch.nn.functional.linear(hidden, weight).argmax(-1)
        torch.testing.assert_close(kernel(hidden), expected)
