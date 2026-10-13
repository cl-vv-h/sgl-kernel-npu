"""Top-k/top-p renormalization using fused filtering and RN restoration.

Experimental top-p scan/filter fusion retains native probability-sort ties
and RN division. Tiled cumulative rounding can differ from native cumsum.
The INT32 probability-merge implementation is diagnostic only, never selected
by the public API. No environment-controlled dispatch.
"""

from typing import Union

import torch
import triton
import triton.language as tl
from sgl_kernel_npu.utils.triton_utils import get_device_properties


def _prepare_probs(probs: torch.Tensor) -> torch.Tensor:
    if probs.ndim != 2:
        raise ValueError("probs must have shape [batch, vocab]")
    if probs.dtype != torch.float32:
        raise ValueError("probs must use torch.float32")
    if probs.device.type != "npu":
        raise ValueError("probs must be on an NPU")
    if not 1 <= probs.shape[1] <= 2**23:
        raise ValueError("probs must have vocab_size in [1, 2**23] for exact FP32 keys")
    return probs.contiguous()


def _as_batch_threshold(
    value: Union[torch.Tensor, int, float],
    probs: torch.Tensor,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        return torch.full((probs.shape[0],), value, device=probs.device, dtype=dtype)
    if value.ndim == 0:
        return value.to(device=probs.device, dtype=dtype).expand(probs.shape[0])
    if value.ndim != 1 or value.shape[0] != probs.shape[0]:
        raise ValueError(f"{name} must be a scalar or have shape [batch]")
    return value.to(device=probs.device, dtype=dtype)


@triton.jit
def _filter_encode_kernel(
    values,
    indices,
    cumulative,
    thresholds,
    keys,
    R: tl.constexpr,
    V: tl.constexpr,
    NB: tl.constexpr,
    THRESHOLD_STRIDE: tl.constexpr,
    TOP_P: tl.constexpr,
    BLOCK: tl.constexpr,
):
    for flat in tl.range(tl.program_id(0), R * NB, tl.num_programs(0)):
        row = (flat // NB).to(tl.int64)
        offsets = (flat % NB) * BLOCK + tl.arange(0, BLOCK)
        valid = offsets < V
        address = row * V + offsets
        value = tl.load(values + address, valid, other=0.0)
        token = tl.load(indices + address, valid, other=0).to(tl.float32)
        if TOP_P:
            threshold = tl.load(thresholds + row * THRESHOLD_STRIDE)
            prefix = tl.load(cumulative + address, valid, other=0.0)
            # Preserve inclusive cumsum minus current value, including its
            # rounding. Do not substitute a different exclusive scan.
            removed = prefix - value > threshold
            value = tl.where(removed, 0.0, value)
            tl.store(values + address, value, valid)
        else:
            threshold = tl.load(thresholds + row * THRESHOLD_STRIDE)
            value = tl.where(offsets >= threshold, 0.0, value)
            tl.store(values + address, value, valid)
        # V <= 2**23 makes every encoded integer exactly representable in FP32.
        key = token * 2.0 + (value != 0.0).to(tl.float32)
        tl.store(keys + address, key, valid)


@triton.jit
def _restore_kernel(
    probs,
    ordered_keys,
    denominator,
    output,
    R: tl.constexpr,
    V: tl.constexpr,
    NB: tl.constexpr,
    BLOCK: tl.constexpr,
):
    for flat in tl.range(tl.program_id(0), R * NB, tl.num_programs(0)):
        row = (flat // NB).to(tl.int64)
        offsets = (flat % NB) * BLOCK + tl.arange(0, BLOCK)
        valid = offsets < V
        address = row * V + offsets
        value = tl.load(probs + address, valid, other=0.0)
        key = tl.load(ordered_keys + address, valid, other=0.0)
        removed = key == offsets.to(tl.float32) * 2.0
        value = tl.where(removed, 0.0, value)
        scale = tl.load(denominator + row)
        # Fixed round-to-nearest division; never replace with rcp * x.
        value = tl.div_rn(value, scale)
        tl.store(output + address, value, valid)


def filter_and_renorm(
    probs, sorted_values, sorted_indices, thresholds, cumulative=None
):
    """Filter native sorted values, restore vocabulary order and divide with RN.

    The caller supplies contiguous NPU FP32 [rows, vocab] tensors, a nonempty
    vocabulary of at most 2**23 entries, and converted/clamped row thresholds.
    cumulative selects top-p; its absence selects top-k. Only sorted_values is
    modified. Keep the native sorted-value reduction order and denominator floor.
    """
    rows, vocab = probs.shape
    if rows == 0:
        return torch.empty_like(probs)
    blocks = triton.cdiv(vocab, 2048)
    grid = (min(rows * blocks, get_device_properties()[1]),)
    keys = torch.empty_like(sorted_values)
    _filter_encode_kernel[grid](
        sorted_values,
        sorted_indices,
        cumulative if cumulative is not None else sorted_values,
        thresholds,
        keys,
        rows,
        vocab,
        blocks,
        thresholds.stride(0),
        cumulative is not None,
        2048,
    )
    denominator = sorted_values.sum(dim=-1, keepdim=True).clamp_min_(1e-20)
    # Sort exact FP32 token keys; discard sort indices. This avoids the slow
    # full-vocabulary scatter while retaining the native probability-sort ties.
    ordered_keys = keys.sort(dim=-1).values
    output = torch.empty_like(probs)
    _restore_kernel[grid](
        probs,
        ordered_keys,
        denominator,
        output,
        rows,
        vocab,
        blocks,
        2048,
    )
    return output


@triton.jit
def _probability_pair_sort(values, tokens, LOG_TILE: tl.constexpr):
    # Total order: decreasing probability, increasing token ID for equal values.
    # No probability quantization, float/int bitcasts or packed probability keys.
    lane = tl.arange(0, 1 << LOG_TILE)
    for stage in tl.static_range(1, LOG_TILE + 1):
        for step in tl.static_range(stage - 1, -1, -1):
            partner = lane ^ (1 << step)
            other_value = tl.gather(values, partner, 0)
            other_token = tl.gather(tokens, partner, 0)
            other_before = (other_value > values) | (
                (other_value == values) & (other_token < tokens)
            )
            self_before = (values > other_value) | (
                (values == other_value) & (tokens < other_token)
            )
            want_before = ((lane & (1 << stage)) == 0) == ((lane & (1 << step)) == 0)
            swap = tl.where(want_before, other_before, self_before)
            values = tl.where(swap, other_value, values)
            tokens = tl.where(swap, other_token, tokens)
    return values, tokens


@triton.jit(do_not_specialize=["ROWS"])
def _probability_sort_tiles_kernel(
    probs,
    values,
    indices,
    totals,
    ROWS,
    VOCAB: tl.constexpr,
    TILES: tl.constexpr,
    LOG_TILE: tl.constexpr,
    EMIT_TOTALS: tl.constexpr,
):
    BLOCK: tl.constexpr = 1 << LOG_TILE
    lane = tl.arange(0, BLOCK)
    for flat in tl.range(tl.program_id(0), ROWS * TILES, tl.num_programs(0)):
        row = (flat // TILES).to(tl.int64)
        position = (flat % TILES) * BLOCK + lane
        valid = position < VOCAB
        value = tl.load(probs + row * VOCAB + position, valid, other=-float("inf"))
        token = tl.where(valid, position, 2147483647).to(tl.int32)
        value, token = _probability_pair_sort(value, token, LOG_TILE)
        tl.store(values + row * VOCAB + position, value, valid)
        tl.store(indices + row * VOCAB + position, token, valid)
        if EMIT_TOTALS:
            tl.store(totals + flat, tl.sum(tl.where(valid, value, 0.0), 0))


@triton.jit(do_not_specialize=["ROWS"])
def _probability_merge_kernel(
    in_values,
    in_indices,
    out_values,
    out_indices,
    totals,
    ROWS,
    VOCAB: tl.constexpr,
    TILES: tl.constexpr,
    RUN: tl.constexpr,
    LOG_TILE: tl.constexpr,
    STEPS: tl.constexpr,
    EMIT_TOTALS: tl.constexpr,
):
    BLOCK: tl.constexpr = 1 << LOG_TILE
    lane = tl.arange(0, BLOCK)
    for flat in tl.range(tl.program_id(0), ROWS * TILES, tl.num_programs(0)):
        row = (flat // TILES).to(tl.int64)
        start = (flat % TILES) * BLOCK
        pair = (start // (2 * RUN)) * (2 * RUN)
        na = tl.minimum(RUN, VOCAB - pair)
        nb = tl.maximum(0, tl.minimum(RUN, VOCAB - pair - RUN))
        diagonal = start - pair
        lo = tl.maximum(0, diagonal - nb)
        hi = tl.minimum(diagonal, na)
        # Global merge-path probe; ties use the explicit token-ID order.
        for _ in range(STEPS):
            active = lo < hi
            i = (lo + hi) // 2
            j = diagonal - i
            ai = row * VOCAB + pair + i
            bi = row * VOCAB + pair + RUN + j - 1
            av = tl.load(in_values + ai, active & (i < na), other=-float("inf"))
            at = tl.load(in_indices + ai, active & (i < na), other=2147483647)
            bv = tl.load(in_values + bi, active & (j > 0), other=float("inf"))
            bt = tl.load(in_indices + bi, active & (j > 0), other=-1)
            advance = active & (j > 0) & ((av > bv) | ((av == bv) & (at < bt)))
            lo = tl.where(advance, i + 1, lo)
            hi = tl.where(active & ~advance, i, hi)
        a_start = lo
        b_start = diagonal - lo
        an = tl.minimum(BLOCK, na - a_start)
        bn = tl.minimum(BLOCK, nb - b_start)
        a_address = row * VOCAB + pair + a_start + lane
        b_address = row * VOCAB + pair + RUN + b_start + lane
        # Each suffix is loaded to UB in a contiguous segment. Output ranks are
        # then located within UB; there is no per-token indexed global gather.
        av = tl.load(in_values + a_address, lane < an, other=-float("inf"))
        at = tl.load(in_indices + a_address, lane < an, other=2147483647)
        bv = tl.load(in_values + b_address, lane < bn, other=-float("inf"))
        bt = tl.load(in_indices + b_address, lane < bn, other=2147483647)
        local_lo = tl.maximum(0, lane - bn)
        local_hi = tl.minimum(lane, an)
        for _ in range(LOG_TILE + 1):
            active = local_lo < local_hi
            i = (local_lo + local_hi) // 2
            j = lane - i
            ai = tl.minimum(i, BLOCK - 1).to(tl.int32)
            bi = tl.maximum(0, tl.minimum(j - 1, BLOCK - 1)).to(tl.int32)
            a_value = tl.gather(av, ai, 0)
            a_token = tl.gather(at, ai, 0)
            a_value = tl.where(i < an, a_value, -float("inf"))
            a_token = tl.where(i < an, a_token, 2147483647)
            b_value = tl.gather(bv, bi, 0)
            b_token = tl.gather(bt, bi, 0)
            advance = (
                active
                & (j > 0)
                & ((a_value > b_value) | ((a_value == b_value) & (a_token < b_token)))
            )
            local_lo = tl.where(advance, i + 1, local_lo)
            local_hi = tl.where(active & ~advance, i, local_hi)
        i = local_lo
        j = lane - i
        ai = tl.minimum(i, BLOCK - 1).to(tl.int32)
        bi = tl.minimum(j, BLOCK - 1).to(tl.int32)
        a_value = tl.gather(av, ai, 0)
        a_token = tl.gather(at, ai, 0)
        b_value = tl.gather(bv, bi, 0)
        b_token = tl.gather(bt, bi, 0)
        choose_a = (i < an) & (
            (j >= bn)
            | (a_value > b_value)
            | ((a_value == b_value) & (a_token < b_token))
        )
        value = tl.where(choose_a, a_value, b_value)
        token = tl.where(choose_a, a_token, b_token)
        position = start + lane
        valid = position < VOCAB
        tl.store(out_values + row * VOCAB + position, value, valid)
        tl.store(out_indices + row * VOCAB + position, token, valid)
        if EMIT_TOTALS:
            tl.store(totals + flat, tl.sum(tl.where(valid, value, 0.0), 0))


def _sort_probabilities_int32(probs, tile=512, emit_totals=False):
    """Experimental descending FP32 sort, with INT32 IDs and deterministic ties.

    Equal probabilities use increasing token ID. This is not a promise to
    reproduce CANN stable=False identities inside a tie group. Input probabilities
    must be finite and nonnegative. All addresses use INT64 arithmetic.
    """
    if tile not in (256, 512, 1024):
        raise ValueError("probability tile must be 256, 512 or 1024")
    rows, vocab = probs.shape
    blocks = triton.cdiv(vocab, tile)
    values = torch.empty_like(probs)
    indices = torch.empty(probs.shape, dtype=torch.int32, device=probs.device)
    totals = torch.empty(
        (rows, blocks) if emit_totals else (1,), dtype=probs.dtype, device=probs.device
    )
    if rows == 0:
        return values, indices, totals if emit_totals else None
    log_tile = tile.bit_length() - 1
    grid = (min(rows * blocks, get_device_properties()[1]),)
    _probability_sort_tiles_kernel[grid](
        probs,
        values,
        indices,
        totals,
        rows,
        vocab,
        blocks,
        log_tile,
        emit_totals and blocks == 1,
    )
    if blocks > 1:
        scratch_values = torch.empty_like(values)
        scratch_indices = torch.empty_like(indices)
        run = tile
        while run < vocab:
            _probability_merge_kernel[grid](
                values,
                indices,
                scratch_values,
                scratch_indices,
                totals,
                rows,
                vocab,
                blocks,
                run,
                log_tile,
                run.bit_length(),
                emit_totals and run * 2 >= vocab,
            )
            values, scratch_values = scratch_values, values
            indices, scratch_indices = scratch_indices, indices
            run *= 2
    return values, indices, totals if emit_totals else None


@triton.jit(do_not_specialize=["ROWS"])
def _probability_tile_totals_kernel(
    values,
    totals,
    ROWS,
    VOCAB: tl.constexpr,
    TILES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    lane = tl.arange(0, BLOCK)
    for flat in tl.range(tl.program_id(0), ROWS * TILES, tl.num_programs(0)):
        row = (flat // TILES).to(tl.int64)
        position = (flat % TILES) * BLOCK + lane
        value = tl.load(values + row * VOCAB + position, position < VOCAB, other=0.0)
        tl.store(totals + flat, tl.sum(value, 0))


@triton.jit(do_not_specialize=["ROWS"])
def _scan_filter_encode_kernel(
    values,
    indices,
    carries,
    thresholds,
    keys,
    ROWS,
    VOCAB: tl.constexpr,
    TILES: tl.constexpr,
    THRESHOLD_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    lane = tl.arange(0, BLOCK)
    for flat in tl.range(tl.program_id(0), ROWS * TILES, tl.num_programs(0)):
        row = (flat // TILES).to(tl.int64)
        tile = flat % TILES
        position = tile * BLOCK + lane
        valid = position < VOCAB
        address = row * VOCAB + position
        value = tl.load(values + address, valid, other=0.0)
        token = tl.load(indices + address, valid, other=0).to(tl.float32)
        carry = tl.load(carries + row * TILES + tile - 1, tile > 0, other=0.0)
        prefix = tl.cumsum(value, 0) + carry
        threshold = tl.load(thresholds + row * THRESHOLD_STRIDE)
        retained = tl.where(prefix - value > threshold, 0.0, value)
        tl.store(values + address, retained, valid)
        tl.store(keys + address, token * 2.0 + (retained != 0.0).to(tl.float32), valid)


def _scan_filter_and_encode(values, indices, thresholds, tile):
    """Fuse tiled scan/filter; only values is mutated. Sorted input ties are kept."""
    rows, vocab = values.shape
    blocks = triton.cdiv(vocab, tile)
    grid = (min(rows * blocks, get_device_properties()[1]),)
    totals = torch.empty((rows, blocks), dtype=values.dtype, device=values.device)
    _probability_tile_totals_kernel[grid](values, totals, rows, vocab, blocks, tile)
    # Only a small [rows, tiles] inclusive carry tensor is materialized.
    carries = totals.cumsum(-1)
    keys = torch.empty_like(values)
    _scan_filter_encode_kernel[grid](
        values,
        indices,
        carries,
        thresholds,
        keys,
        rows,
        vocab,
        blocks,
        thresholds.stride(0),
        tile,
    )
    return keys


def _top_p_renorm_prob_scan_fused(probs, top_ps, tile=2048):
    """Experimental tiled scan/filter; cumulative rounding differs from native.

    Uses native probability sorting, including native tie identities. RN division
    and native masked-value reduction are unchanged. Native-cumsum equivalence
    must be measured; this is an experimental branch, not the stable PR.
    """
    probs = _prepare_probs(probs)
    top_ps = _as_batch_threshold(top_ps, probs, probs.dtype, "top_ps").clamp(0, 1)
    rows, vocab = probs.shape
    if rows == 0:
        return torch.empty_like(probs)
    if tile not in (512, 1024, 2048):
        raise ValueError("scan tile must be 512, 1024 or 2048")
    values, indices = probs.sort(dim=-1, descending=True)
    keys = _scan_filter_and_encode(values, indices, top_ps, tile)
    blocks = triton.cdiv(vocab, tile)
    grid = (min(rows * blocks, get_device_properties()[1]),)
    denominator = values.sum(-1, keepdim=True).clamp_min_(1e-20)
    ordered_keys = keys.sort(-1).values
    output = torch.empty_like(probs)
    _restore_kernel[grid](
        probs, ordered_keys, denominator, output, rows, vocab, blocks, tile
    )
    return output


def top_k_renorm_prob(
    probs: torch.Tensor,
    top_ks: Union[torch.Tensor, int],
) -> torch.Tensor:
    """Keep top-k tokens and renormalize NPU FP32 probabilities with RN division."""
    probs = _prepare_probs(probs)
    top_ks = _as_batch_threshold(top_ks, probs, torch.long, "top_ks").clamp(
        min=1, max=probs.shape[1]
    )
    if probs.shape[0] == 0:
        return torch.empty_like(probs)
    sorted_probs, sorted_indices = probs.sort(dim=-1, descending=True)
    return filter_and_renorm(probs, sorted_probs, sorted_indices, top_ks)


def _top_p_renorm_prob_sort_int32(probs, top_ps, tile=512):
    """Benchmark candidate: INT32 first sort with native cumsum and key sort.

    Equal probabilities use token-ID order, not native unstable tie identities.
    This function is deliberately separate from the service default.
    """
    probs = _prepare_probs(probs)
    top_ps = _as_batch_threshold(top_ps, probs, probs.dtype, "top_ps").clamp(0, 1)
    if probs.shape[0] == 0:
        return torch.empty_like(probs)
    values, indices, _ = _sort_probabilities_int32(probs, tile)
    return filter_and_renorm(probs, values, indices, top_ps, values.cumsum(-1))


def top_p_renorm_prob(
    probs: torch.Tensor,
    top_ps: Union[torch.Tensor, float],
) -> torch.Tensor:
    """Experimental fused tiled prefix scan/filter with native sort and RN.

    Sorting identities and the cutoff predicate are unchanged. Floating prefix
    rounding differs from native cumsum; use the benchmark's strict V1 gate
    before deploying. No multi-level probability/key merge is used here.
    """
    return _top_p_renorm_prob_scan_fused(probs, top_ps)
