# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 by FlashInfer team.
# Modifications: NPU Triton implementation using partitioned reductions,
# four-bit radix passes, full-row rescans and round-to-nearest normalization.
# Reference: flashinfer/include/flashinfer/air_top_p.cuh at
# c94c1a9e4bf151e56080168ac880ebf0651af317. See docs/developer_guide/sampling_air_top_p.md.
"""Experimental AIR-style top-p renormalization for FP32 NPU probabilities.

This is a weighted radix SELECT, not a radix sort. Each pass refines only the
selected bit prefix. Partial histograms have a unique writer; no scatter or
atomic updates are required. This initial port rescans input each pass rather
than compacting the boundary bucket into CUDA-style scratch buffers.

Inputs must be finite, nonnegative probabilities. All ties at the cutoff are
kept. The target mass is top_p * row_sum; this is not bitwise V1 semantics.
"""

import torch
import triton
import triton.language as tl
from sgl_kernel_npu.utils.triton_utils import get_device_properties

_RADIX_BITS = 4
_BINS = 1 << _RADIX_BITS


@triton.jit
def _air_histogram_kernel(
    probs,
    prefix,
    histogram,
    ROWS: tl.constexpr,
    VOCAB: tl.constexpr,
    TILES: tl.constexpr,
    SHIFT: tl.constexpr,
    FIRST: tl.constexpr,
    BLOCK: tl.constexpr,
):
    bins = tl.arange(0, 16)
    for flat in tl.range(tl.program_id(0), ROWS * TILES, tl.num_programs(0)):
        row = (flat // TILES).to(tl.int64)
        offset = (flat % TILES) * BLOCK + tl.arange(0, BLOCK)
        valid = offset < VOCAB
        value = tl.load(probs + row * VOCAB + offset, valid, other=0.0)
        # For nonnegative FP32, reversed bits order values from high to low.
        key = value.to(tl.uint32, bitcast=True) ^ 0x7FFFFFFF
        if not FIRST:
            previous = tl.load(prefix + row).to(tl.uint32)
            valid = valid & ((key >> (SHIFT + 4)) == (previous >> (SHIFT + 4)))
        bucket = (key >> SHIFT) & 15
        # Small, dense local histogram. Do not substitute a global atomic_add
        # per token: concentrated probability buckets would serialize it.
        mass = tl.sum(
            tl.where(
                valid[None, :] & (bins[:, None] == bucket[None, :]), value[None, :], 0.0
            ),
            1,
        )
        tl.store(histogram + flat.to(tl.int64) * 16 + bins, mass)


@triton.jit
def _air_select_kernel(
    histogram,
    prefix,
    remaining,
    top_ps,
    TILES: tl.constexpr,
    PAD_TILES: tl.constexpr,
    P_STRIDE: tl.constexpr,
    SHIFT: tl.constexpr,
    FIRST: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    tiles = tl.arange(0, PAD_TILES)
    bins = tl.arange(0, 16)
    masses = tl.load(
        histogram + (row * TILES + tiles[:, None]) * 16 + bins[None, :],
        tiles[:, None] < TILES,
        other=0.0,
    )
    hist = tl.sum(masses, 0)
    if FIRST:
        target = tl.load(top_ps + row * P_STRIDE) * tl.sum(hist, 0)
        old_prefix = tl.full((), 0, tl.uint32)
    else:
        target = tl.load(remaining + row)
        old_prefix = tl.load(prefix + row).to(tl.uint32)
    # This prefix sum has 16 entries, never vocab_size entries.
    cumulative = tl.cumsum(hist, 0)
    chosen = tl.min(tl.where((hist > 0.0) & (cumulative >= target), bins, 16), 0)
    last = tl.max(tl.where(hist > 0.0, bins, -1), 0)
    # A rounding residual slightly above this bucket's total selects its last
    # occupied sub-bucket. All-zero rows converge to -0.0 and normalize to zero.
    chosen = tl.where(chosen < 16, chosen, tl.where(last >= 0, last, 15))
    before = tl.sum(tl.where(bins < chosen, hist, 0.0), 0)
    tl.store(remaining + row, tl.maximum(target - before, 0.0))
    tl.store(prefix + row, old_prefix | (chosen.to(tl.uint32) << SHIFT))


@triton.jit
def _air_kept_mass_kernel(
    probs,
    prefix,
    top_ps,
    partial_sums,
    ROWS: tl.constexpr,
    VOCAB: tl.constexpr,
    TILES: tl.constexpr,
    P_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    for flat in tl.range(tl.program_id(0), ROWS * TILES, tl.num_programs(0)):
        row = (flat // TILES).to(tl.int64)
        offset = (flat % TILES) * BLOCK + tl.arange(0, BLOCK)
        valid = offset < VOCAB
        key = tl.load(prefix + row).to(tl.uint32)
        cutoff = (key ^ 0x7FFFFFFF).to(tl.float32, bitcast=True)
        value = tl.load(probs + row * VOCAB + offset, valid, other=0.0)
        # Explicit p=1 avoids dropping tiny values through prefix-sum rounding.
        keep = (value >= cutoff) | (tl.load(top_ps + row * P_STRIDE) >= 1.0)
        tl.store(partial_sums + flat, tl.sum(tl.where(valid & keep, value, 0.0), 0))


@triton.jit
def _air_denominator_kernel(
    partial_sums, denominator, TILES: tl.constexpr, PAD_TILES: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    tiles = tl.arange(0, PAD_TILES)
    values = tl.load(partial_sums + row * TILES + tiles, tiles < TILES, other=0.0)
    tl.store(denominator + row, tl.maximum(tl.sum(values, 0), 1e-20))


@triton.jit
def _air_apply_kernel(
    probs,
    prefix,
    top_ps,
    denominator,
    output,
    ROWS: tl.constexpr,
    VOCAB: tl.constexpr,
    TILES: tl.constexpr,
    P_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    for flat in tl.range(tl.program_id(0), ROWS * TILES, tl.num_programs(0)):
        row = (flat // TILES).to(tl.int64)
        offset = (flat % TILES) * BLOCK + tl.arange(0, BLOCK)
        valid = offset < VOCAB
        key = tl.load(prefix + row).to(tl.uint32)
        cutoff = (key ^ 0x7FFFFFFF).to(tl.float32, bitcast=True)
        value = tl.load(probs + row * VOCAB + offset, valid, other=0.0)
        keep = (value >= cutoff) | (tl.load(top_ps + row * P_STRIDE) >= 1.0)
        value = tl.where(keep, value, 0.0)
        value = tl.div_rn(value, tl.load(denominator + row))
        tl.store(output + row * VOCAB + offset, value, valid)


def air_top_p_renorm(probs, top_ps, block_size=512, *, stage_hook=None):
    """Return a full [rows, vocab] tensor in vocabulary order, using RN division.

    top_ps is a clamped FP32 vector [rows] (stride zero is supported). No tensor
    values are inspected on the host. stage_hook(name, begin) is diagnostics
    only; leave unset in service and in end-to-end latency measurements.
    """
    if probs.ndim != 2 or probs.dtype != torch.float32 or not probs.is_contiguous():
        raise ValueError("AIR requires a contiguous FP32 [rows, vocab] tensor")
    if probs.device.type != "npu":
        raise ValueError("AIR requires NPU tensors")
    rows, vocab = probs.shape
    if vocab < 1 or vocab > 2**23:
        raise ValueError("AIR currently supports vocab in [1, 2**23]")
    if (
        top_ps.shape != (rows,)
        or top_ps.dtype != torch.float32
        or top_ps.device != probs.device
    ):
        raise ValueError("top_ps must be FP32 [rows] on the same NPU")
    if block_size not in (256, 512, 1024):
        raise ValueError("AIR block_size must be 256, 512 or 1024")
    output = torch.empty_like(probs)
    if not rows:
        return output
    tiles = triton.cdiv(vocab, block_size)
    # The selection reduction is deliberately bounded for Ascend UB usage.
    # Public dispatch falls back to V1 for larger vocabularies/configurations.
    if triton.next_power_of_2(tiles) > 1024:
        raise ValueError("AIR requires ceil(vocab / block_size) <= 1024")
    histogram = torch.empty(
        (rows, tiles, _BINS), device=probs.device, dtype=torch.float32
    )
    prefix = torch.empty((rows,), device=probs.device, dtype=torch.int32)
    remaining = torch.empty((rows,), device=probs.device, dtype=torch.float32)
    partial_sums = torch.empty((rows, tiles), device=probs.device, dtype=torch.float32)
    denominator = torch.empty((rows,), device=probs.device, dtype=torch.float32)
    grid = (min(rows * tiles, get_device_properties()[1]),)
    p_stride = top_ps.stride(0)

    def launch(name, kernel, launch_grid, *args):
        if stage_hook is not None:
            stage_hook(name, True)
        kernel[launch_grid](*args)
        if stage_hook is not None:
            stage_hook(name, False)

    for shift in range(28, -1, -4):
        launch(
            f"histogram_{shift}",
            _air_histogram_kernel,
            grid,
            probs,
            prefix,
            histogram,
            rows,
            vocab,
            tiles,
            shift,
            shift == 28,
            block_size,
        )
        launch(
            f"select_{shift}",
            _air_select_kernel,
            (rows,),
            histogram,
            prefix,
            remaining,
            top_ps,
            tiles,
            triton.next_power_of_2(tiles),
            p_stride,
            shift,
            shift == 28,
        )
    launch(
        "kept_mass",
        _air_kept_mass_kernel,
        grid,
        probs,
        prefix,
        top_ps,
        partial_sums,
        rows,
        vocab,
        tiles,
        p_stride,
        block_size,
    )
    launch(
        "denominator",
        _air_denominator_kernel,
        (rows,),
        partial_sums,
        denominator,
        tiles,
        triton.next_power_of_2(tiles),
    )
    launch(
        "apply_rn",
        _air_apply_kernel,
        grid,
        probs,
        prefix,
        top_ps,
        denominator,
        output,
        rows,
        vocab,
        tiles,
        p_stride,
        block_size,
    )
    return output
