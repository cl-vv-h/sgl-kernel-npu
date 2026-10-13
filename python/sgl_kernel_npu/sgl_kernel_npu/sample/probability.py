"""Top-k/top-p renormalization using fused filtering and RN restoration.

The only implementation is the values-only packed-key merge path for NPU
FP32 probabilities. There is no environment-controlled dispatch or scatter fallback.
"""

from typing import Union

import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al
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
    SORT_KEYS: tl.constexpr = False,
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
        if SORT_KEYS:
            key = al.sort(tl.where(valid, key, float("inf")), descending=False, dim=0)
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


@triton.jit
def _sort_key_tiles_kernel(
    keys,
    output,
    R: tl.constexpr,
    V: tl.constexpr,
    NB: tl.constexpr,
    BLOCK: tl.constexpr,
):
    for flat in tl.range(tl.program_id(0), R * NB, tl.num_programs(0)):
        row = (flat // NB).to(tl.int64)
        offsets = (flat % NB) * BLOCK + tl.arange(0, BLOCK)
        value = tl.load(keys + row * V + offsets, offsets < V, other=float("inf"))
        value = al.sort(value, descending=False, dim=0)
        tl.store(output + row * V + offsets, value, offsets < V)


@triton.jit
def _merge_key_tiles_kernel(
    source,
    destination,
    probs,
    denominator,
    R: tl.constexpr,
    V: tl.constexpr,
    NB: tl.constexpr,
    RUN: tl.constexpr,
    BLOCK: tl.constexpr,
    STEPS: tl.constexpr,
    FINAL_RESTORE: tl.constexpr,
):
    # Merge-path partitions an output tile into two already sorted runs.
    # Input loads and output stores are contiguous; only partition probes are
    # scalar indexed loads. No token-indexed global scatter is used.
    for flat in tl.range(tl.program_id(0), R * NB, tl.num_programs(0)):
        row = (flat // NB).to(tl.int64)
        start = (flat % NB) * BLOCK
        pair = (start // (2 * RUN)) * (2 * RUN)
        na = tl.minimum(RUN, V - pair)
        nb = tl.maximum(0, tl.minimum(RUN, V - pair - RUN))
        diagonal = start - pair
        lo = tl.maximum(0, diagonal - nb)
        hi = tl.minimum(diagonal, na)
        # Find the smallest i with A[i] >= B[diagonal-i-1]. Packed keys
        # are unique, so comparison ties cannot occur between the two runs.
        for _ in range(STEPS):
            active = lo < hi
            i = (lo + hi) // 2
            j = diagonal - i
            a = tl.load(
                source + row * V + pair + i, active & (i < na), other=float("inf")
            )
            b = tl.load(
                source + row * V + pair + RUN + j - 1,
                active & (j > 0),
                other=-float("inf"),
            )
            advance = active & (a < b)
            lo = tl.where(advance, i + 1, lo)
            hi = tl.where(active & ~advance, i, hi)
        a_start = lo
        b_start = diagonal - lo
        # The first BLOCK items of each suffix contain the next BLOCK merged
        # items. Sort these bounded local tensors, then retain their first half.
        positions = tl.arange(0, 2 * BLOCK)
        left = positions < BLOCK
        lane = positions % BLOCK
        index = tl.where(left, pair + a_start + lane, pair + RUN + b_start + lane)
        valid = tl.where(left, a_start + lane < na, b_start + lane < nb)
        value = tl.load(source + row * V + index, valid, other=float("inf"))
        merged = al.sort(value, descending=False, dim=0)
        # Only the first half is stored, with disjoint, aligned output tiles.
        offsets = start + positions
        store_mask = (positions < BLOCK) & (offsets < V)
        if FINAL_RESTORE:
            probability = tl.load(probs + row * V + offsets, store_mask, other=0.0)
            probability = tl.where(
                merged == offsets.to(tl.float32) * 2.0, 0.0, probability
            )
            scale = tl.load(denominator + row)
            merged = tl.div_rn(probability, scale)
        tl.store(destination + row * V + offsets, merged, store_mask)


def _merge_packed_key_runs(keys, tile, probs=None, denominator=None):
    """Merge sorted FP32 key tiles, optionally restoring probabilities at the end.

    Caller supplies contiguous keys and disjoint tiles. No host readback,
    atomics, default-stream switch or cached shared workspace is used.
    """
    rows, vocab = keys.shape
    blocks = triton.cdiv(vocab, tile)
    grid = (min(rows * blocks, get_device_properties()[1]),)
    restore = probs is not None
    if blocks == 1:
        if not restore:
            return keys
        output = torch.empty_like(probs)
        _restore_kernel[grid](
            probs, keys, denominator, output, rows, vocab, blocks, tile
        )
        return output
    # Ping-pong buffers are owned by this invocation/graph capture. The keys
    # buffer may be reused only after the preceding merge reads finish.
    scratch = torch.empty_like(keys)
    source, destination = keys, scratch
    run = tile
    while run < vocab:
        final = run * 2 >= vocab
        if final and restore:
            destination = torch.empty_like(probs)
        _merge_key_tiles_kernel[grid](
            source,
            destination,
            probs if restore else keys,
            denominator if restore else keys,
            rows,
            vocab,
            blocks,
            run,
            tile,
            run.bit_length(),
            final and restore,
        )
        source, destination = destination, source
        run *= 2
    return source


def _sort_packed_keys(keys, tile=2048):
    """Values-only ascending sort for unique finite FP32 packed token keys.

    Internal contract: keys = 2*token_id + keep, one key per token. This helper
    does not replace general probability sorting. tile is a compile-time test
    parameter, not an environment-controlled implementation selector.
    """
    if tile not in (512, 1024, 2048):
        raise ValueError("key tile must be 512, 1024 or 2048")
    rows, vocab = keys.shape
    if rows == 0:
        return torch.empty_like(keys)
    tiles = torch.empty_like(keys)
    blocks = triton.cdiv(vocab, tile)
    grid = (min(rows * blocks, get_device_properties()[1]),)
    _sort_key_tiles_kernel[grid](keys, tiles, rows, vocab, blocks, tile)
    return _merge_packed_key_runs(tiles, tile)


def filter_and_renorm(
    probs, sorted_values, sorted_indices, thresholds, cumulative=None, key_tile=2048
):
    """Filter native sorted values, restore vocabulary order and divide with RN.

    The caller supplies contiguous NPU FP32 [rows, vocab] tensors, a nonempty
    vocabulary of at most 2**23 entries, and converted/clamped row thresholds.
    cumulative selects top-p; its absence selects top-k. Only sorted_values is
    modified. Keep the native sorted-value reduction order and denominator floor.
    """
    if key_tile not in (512, 1024, 2048):
        raise ValueError("key tile must be 512, 1024 or 2048")
    rows, vocab = probs.shape
    if rows == 0:
        return torch.empty_like(probs)
    blocks = triton.cdiv(vocab, key_tile)
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
        key_tile,
        True,
    )
    denominator = sorted_values.sum(dim=-1, keepdim=True).clamp_min_(1e-20)
    # Filtering already sorted each key tile. Final merge restores probabilities
    # directly, avoiding full INT64 indices and a final ordered-key round trip.
    return _merge_packed_key_runs(keys, key_tile, probs, denominator)


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


def top_p_renorm_prob(
    probs: torch.Tensor,
    top_ps: Union[torch.Tensor, float],
) -> torch.Tensor:
    """Keep the native sorted nucleus and renormalize NPU FP32 probabilities.

    Selection uses inclusive cumsum minus the current value, preserving the
    original rounding and tie behavior. Both endpoints of top_p are supported.
    """
    probs = _prepare_probs(probs)
    top_ps = _as_batch_threshold(top_ps, probs, probs.dtype, "top_ps").clamp(
        min=0.0, max=1.0
    )
    if probs.shape[0] == 0:
        return torch.empty_like(probs)
    sorted_probs, sorted_indices = probs.sort(dim=-1, descending=True)
    cumulative_probs = sorted_probs.cumsum(dim=-1)
    return filter_and_renorm(
        probs, sorted_probs, sorted_indices, top_ps, cumulative_probs
    )
