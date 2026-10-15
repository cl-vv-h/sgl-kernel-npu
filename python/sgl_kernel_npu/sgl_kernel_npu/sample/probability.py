"""Top-k/top-p renormalization using fused filtering and RN restoration.

The default is the V1 packed-key path for NPU FP32 probabilities. Independent
startup switches enable experimental boundary restoration and tiled scan fusion.
"""

import os
from typing import Union

import torch
import triton
import triton.language as tl
from sgl_kernel_npu.utils.triton_utils import get_device_properties

# Read once at import: change these before starting/restarting the service.
_BOUNDARY_RESTORE = (
    os.environ.get("SGL_KERNEL_NPU_SAMPLING_BOUNDARY_RESTORE", "0") == "1"
)
_SCAN_FUSION = os.environ.get("SGL_KERNEL_NPU_SAMPLING_SCAN_FUSION", "0") == "1"


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


def _filter_and_renorm_v1(
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


@triton.jit(do_not_specialize=["ROWS"])
def _tile_totals_kernel(
    values,
    totals,
    ROWS,
    V: tl.constexpr,
    NB: tl.constexpr,
    BLOCK: tl.constexpr,
):
    for flat in tl.range(tl.program_id(0), ROWS * NB, tl.num_programs(0)):
        row = (flat // NB).to(tl.int64)
        position = (flat % NB) * BLOCK + tl.arange(0, BLOCK)
        value = tl.load(values + row * V + position, position < V, other=0.0)
        tl.store(totals + flat, tl.sum(value, 0))


@triton.jit(do_not_specialize=["ROWS"])
def _experimental_filter_kernel(
    values,
    indices,
    prefixes,
    thresholds,
    auxiliary,
    ROWS,
    V: tl.constexpr,
    NB: tl.constexpr,
    THRESHOLD_STRIDE: tl.constexpr,
    TOP_P: tl.constexpr,
    SCAN_FUSION: tl.constexpr,
    BOUNDARY_RESTORE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    lane = tl.arange(0, BLOCK)
    for flat in tl.range(tl.program_id(0), ROWS * NB, tl.num_programs(0)):
        row = (flat // NB).to(tl.int64)
        tile = flat % NB
        position = tile * BLOCK + lane
        valid = position < V
        address = row * V + position
        value = tl.load(values + address, valid, other=0.0)
        threshold = tl.load(thresholds + row * THRESHOLD_STRIDE)
        if TOP_P:
            if SCAN_FUSION:
                carry = tl.load(prefixes + row * NB + tile - 1, tile > 0, other=0.0)
                prefix = tl.cumsum(value, 0) + carry
            else:
                prefix = tl.load(prefixes + address, valid, other=0.0)
            retained = tl.where(prefix - value > threshold, 0.0, value)
        else:
            retained = tl.where(position >= threshold, 0.0, value)
        tl.store(values + address, retained, valid)
        if BOUNDARY_RESTORE:
            # Restoring probabilities above the row's maximum removed value
            # never restores a removed token. The block's minimum positive
            # retained value identifies whether any retained tokens need repair.
            a = tl.min(tl.where(valid & (retained > 0.0), value, float("inf")), 0)
            b = tl.max(
                tl.where(valid & (value > 0.0) & (retained == 0.0), value, 0.0), 0
            )
            # Use aligned, contiguous planes for the metadata reductions.
            # Interleaved pairs produce masked stride-2 copies in BiSheng IR.
            bound_stride: tl.constexpr = triton.next_power_of_2(NB)
            base = row * (2 * bound_stride) + tile
            tl.store(auxiliary + base, a)
            tl.store(auxiliary + base + bound_stride, b)
        else:
            token = tl.load(indices + address, valid, other=0).to(tl.float32)
            key = token * 2.0 + (retained != 0.0).to(tl.float32)
            tl.store(auxiliary + address, key, valid)


@triton.jit(do_not_specialize=["ROWS"])
def _boundary_metadata_kernel(
    bounds,
    metadata,
    ROWS,
    NB: tl.constexpr,
    PAD: tl.constexpr,
):
    tile = tl.arange(0, PAD)
    valid = tile < NB
    for row_index in tl.range(tl.program_id(0), ROWS, tl.num_programs(0)):
        row = row_index.to(tl.int64)
        base = row * (2 * PAD) + tile
        local_a = tl.load(bounds + base, valid, other=float("inf"))
        b = tl.max(tl.load(bounds + base + PAD, valid, other=0.0), 0)
        # A block needs repair iff it contains a positive retained value <= b.
        # When every retained value is above b, begin > end skips row repair.
        needs_correction = valid & (local_a <= b)
        begin = tl.min(tl.where(needs_correction, tile, NB), 0)
        end = tl.max(tl.where(needs_correction, tile + 1, 0), 0)
        tl.store(metadata + row * 4, b)
        tl.store(metadata + row * 4 + 1, begin.to(tl.float32))
        tl.store(metadata + row * 4 + 2, end.to(tl.float32))


@triton.jit(do_not_specialize=["ROWS"])
def _boundary_restore_kernel(
    probs,
    metadata,
    denominator,
    output,
    ROWS,
    V: tl.constexpr,
    NB: tl.constexpr,
    OUTPUT_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    for flat in tl.range(tl.program_id(0), ROWS * NB, tl.num_programs(0)):
        row = (flat // NB).to(tl.int64)
        position = (flat % NB) * BLOCK + tl.arange(0, BLOCK)
        valid = position < V
        cutoff = tl.load(metadata + row * 4)
        value = tl.load(probs + row * V + position, valid, other=0.0)
        # No removed positive probability exceeds cutoff. Only retained values
        # at or below cutoff are missing and will be restored by row repair.
        value = tl.where(value > cutoff, value, 0.0)
        value = tl.div_rn(value, tl.load(denominator + row))
        tl.store(output + row * OUTPUT_STRIDE + position, value, valid)


@triton.jit(do_not_specialize=["ROWS"])
def _boundary_correct_kernel(
    values,
    indices,
    metadata,
    denominator,
    output,
    ROWS,
    V: tl.constexpr,
    OUTPUT_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    lane = tl.arange(0, BLOCK)
    # One program owns each row's indexed writes, including all correction
    # blocks. Do not parallelize these stores into different row-sharing blocks.
    for row_index in tl.range(tl.program_id(0), ROWS, tl.num_programs(0)):
        row = row_index.to(tl.int64)
        begin = tl.load(metadata + row * 4 + 1).to(tl.int32)
        end = tl.load(metadata + row * 4 + 2).to(tl.int32)
        cutoff = tl.load(metadata + row * 4)
        scale = tl.load(denominator + row)
        for tile in tl.range(begin, end):
            position = tile * BLOCK + lane
            valid = position < V
            address = row * V + position
            # Keep the native sort's tie identities. Only positive retained
            # values missed by threshold restoration need indexed loads/stores.
            value = tl.load(values + address, valid, other=0.0)
            needs_correction = valid & (value > 0.0) & (value <= cutoff)
            token = tl.load(indices + address, needs_correction, other=0)
            value = tl.div_rn(value, scale)
            tl.store(output + row * OUTPUT_STRIDE + token, value, needs_correction)


def _renorm_sorted_experimental(
    probs,
    sorted_values,
    sorted_indices,
    thresholds,
    cumulative=None,
    *,
    boundary_restore=False,
    scan_fusion=False,
):
    """Independent experiments on a shared native probability permutation.

    Boundary restoration preserves the supplied selection, including ties and
    non-prefix cutoff decisions caused by rounding. It requires finite,
    nonnegative probabilities. Scan fusion is top-p only and changes FP32
    addition order, so selection equivalence to native cumsum is not assumed.
    """
    if not boundary_restore and not scan_fusion:
        return _filter_and_renorm_v1(
            probs, sorted_values, sorted_indices, thresholds, cumulative
        )
    rows, vocab = probs.shape
    if rows == 0:
        return torch.empty_like(probs)
    block = 2048
    blocks = triton.cdiv(vocab, block)
    grid = (min(rows * blocks, get_device_properties()[1]),)
    row_grid = (min(rows, get_device_properties()[1]),)
    if scan_fusion:
        totals = torch.empty((rows, blocks), device=probs.device, dtype=torch.float32)
        _tile_totals_kernel[grid](sorted_values, totals, rows, vocab, blocks, block)
        prefixes = totals.cumsum(-1)
    else:
        prefixes = cumulative if cumulative is not None else sorted_values
    auxiliary = (
        torch.empty(
            (rows, 2, triton.next_power_of_2(blocks)),
            device=probs.device,
            dtype=torch.float32,
        )
        if boundary_restore
        else torch.empty_like(sorted_values)
    )
    _experimental_filter_kernel[grid](
        sorted_values,
        sorted_indices,
        prefixes,
        thresholds,
        auxiliary,
        rows,
        vocab,
        blocks,
        thresholds.stride(0),
        scan_fusion or cumulative is not None,
        scan_fusion,
        boundary_restore,
        block,
    )
    denominator = sorted_values.sum(-1, keepdim=True).clamp_min_(1e-20)
    if not boundary_restore:
        ordered_keys = auxiliary.sort(-1).values
        output = torch.empty_like(probs)
        _restore_kernel[grid](
            probs, ordered_keys, denominator, output, rows, vocab, blocks, block
        )
        return output
    # Three fields (cutoff, begin, end), with a padded four-element row stride.
    metadata = torch.empty((rows, 4), device=probs.device, dtype=torch.float32)
    _boundary_metadata_kernel[row_grid](
        auxiliary, metadata, rows, blocks, triton.next_power_of_2(blocks)
    )
    # Align row ownership to 32 bytes for the indexed correction stores.
    # The production vocabularies are already aligned; only tail tests copy.
    output_stride = triton.cdiv(vocab, 8) * 8
    storage = torch.empty(
        (rows, output_stride), device=probs.device, dtype=torch.float32
    )
    output = storage[:, :vocab]
    _boundary_restore_kernel[grid](
        probs, metadata, denominator, storage, rows, vocab, blocks, output_stride, block
    )
    _boundary_correct_kernel[row_grid](
        sorted_values,
        sorted_indices,
        metadata,
        denominator,
        storage,
        rows,
        vocab,
        output_stride,
        block,
    )
    return output.contiguous()


def filter_and_renorm(
    probs, sorted_values, sorted_indices, thresholds, cumulative=None
):
    """V1 by default; the boundary switch also supports top-k callers."""
    return _renorm_sorted_experimental(
        probs,
        sorted_values,
        sorted_indices,
        thresholds,
        cumulative,
        boundary_restore=_BOUNDARY_RESTORE,
    )


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

    By default selection uses native inclusive cumsum minus the current value.
    The experimental scan switch changes prefix rounding. Both endpoints of
    top_p are supported, and native probability sorting supplies tie identities.
    """
    probs = _prepare_probs(probs)
    top_ps = _as_batch_threshold(top_ps, probs, probs.dtype, "top_ps").clamp(
        min=0.0, max=1.0
    )
    if probs.shape[0] == 0:
        return torch.empty_like(probs)
    sorted_probs, sorted_indices = probs.sort(dim=-1, descending=True)
    if _SCAN_FUSION:
        return _renorm_sorted_experimental(
            probs,
            sorted_probs,
            sorted_indices,
            top_ps,
            boundary_restore=_BOUNDARY_RESTORE,
            scan_fusion=True,
        )
    cumulative_probs = sorted_probs.cumsum(dim=-1)
    return filter_and_renorm(
        probs, sorted_probs, sorted_indices, top_ps, cumulative_probs
    )
