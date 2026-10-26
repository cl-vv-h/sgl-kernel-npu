import triton
import triton.language as tl


@triton.jit(do_not_specialize=["N", "B"])
def _mix_fused_kernel(
    prefix_ptr,
    bank_ptr,
    cw_ptr,
    out_ptr,
    N,
    B,
    stride_pm,
    stride_bm,
    stride_bb,
    stride_om,
    H: tl.constexpr,
    EPS: tl.constexpr,
    NUM_CORES: tl.constexpr,
    NB: tl.constexpr,
):
    block_size = (N - 1) // NUM_CORES + 1
    pid = tl.program_id(0)
    token_start = pid * block_size
    if token_start >= N:
        return
    token_end = tl.minimum(token_start + block_size, N)

    hidden_offsets = tl.arange(0, H)
    row_offsets = tl.arange(0, NB)
    combined_weight = tl.load(cw_ptr + hidden_offsets).to(tl.float32)

    for token in range(token_start, token_end):
        scores = tl.full([NB], -float("inf"), dtype=tl.float32)
        for row in range(B + 1):
            if row < B:
                value = tl.load(
                    bank_ptr + token * stride_bm + row * stride_bb + hidden_offsets
                ).to(tl.float32)
            else:
                value = tl.load(prefix_ptr + token * stride_pm + hidden_offsets).to(
                    tl.float32
                )
            inverse_rms = tl.rsqrt(tl.sum(value * value) / H + EPS)
            score = tl.sum(value * inverse_rms * combined_weight)
            scores = tl.where(row_offsets == row, score, scores)

        scores_max = tl.max(scores)
        exp_scores = tl.exp(scores - scores_max)
        probabilities = exp_scores / tl.sum(exp_scores)

        output = tl.zeros([H], dtype=tl.float32)
        for row in range(B + 1):
            if row < B:
                value = tl.load(
                    bank_ptr + token * stride_bm + row * stride_bb + hidden_offsets
                ).to(tl.float32)
            else:
                value = tl.load(prefix_ptr + token * stride_pm + hidden_offsets).to(
                    tl.float32
                )
            probability = tl.sum(tl.where(row_offsets == row, probabilities, 0.0))
            output += probability * value

        tl.store(
            out_ptr + token * stride_om + hidden_offsets,
            output.to(out_ptr.dtype.element_ty),
        )
