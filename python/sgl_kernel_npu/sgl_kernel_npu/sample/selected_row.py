"""Exact top-p / target-only sampling for single-child speculative trees.

The caller guarantees tree_topk=1 (no sibling alternatives). Probabilities are
FP32, contiguous, after temperature/softmax and optional top-k renormalization.
The first sort, inclusive cumsum-minus-value, sorted denominator and RN division
match V1. Only the chosen row per request is restored to vocabulary order.
"""

import torch
import triton
import triton.language as tl
from sgl_kernel_npu.sample.probability import _as_batch_threshold
from sgl_kernel_npu.sample.probability_triton import _filter_encode_kernel
from sgl_kernel_npu.sample.tree_speculative_sampling_target_only import (
    _tree_target_only_block_sum_kernel,
    _tree_target_only_sample_kernel,
)
from sgl_kernel_npu.utils.triton_utils import get_device_properties


@triton.jit
def _selected_accept_kernel(
    predicts,
    accept_index,
    accept_token_num,
    candidates,
    retrieve_index,
    next_token,
    coins,
    partials,
    denominator,
    metadata,
    rejected_token,
    THRESHOLD_SINGLE,
    THRESHOLD_ACC,
    D: tl.constexpr,
    DEPTH: tl.constexpr,
    NB: tl.constexpr,
    PAD_NB: tl.constexpr,
):
    req = tl.program_id(0)
    base = req.to(tl.int64) * D
    row = tl.full((), 0, tl.int64)
    output_slot = tl.load(retrieve_index + base).to(tl.int64)
    coin = tl.load(coins + base).to(tl.float32)
    accepted_count = 0
    active = tl.full((), 1, tl.int32)
    rejected = tl.full((), -1, tl.int64)
    tl.store(accept_index + req * DEPTH, output_slot)
    blocks = tl.arange(0, PAD_NB)
    for _depth in range(1, DEPTH):
        if active == 1:
            child = tl.load(next_token + base + row).to(tl.int64)
            if child == -1:
                active = 0
            else:
                parts = tl.load(
                    partials + (base + row) * NB + blocks, blocks < NB, other=0.0
                )
                numerator = tl.max(parts, 0)
                denom = tl.load(denominator + base + row)
                probability = tl.div_rn(numerator, denom)
                token = tl.load(candidates + base + child).to(tl.int64)
                # Identical threshold expression to the full tree kernel.
                if (coin <= probability / THRESHOLD_ACC) | (
                    probability >= THRESHOLD_SINGLE
                ):
                    tl.store(predicts + output_slot, token)
                    accepted_count += 1
                    output_slot = tl.load(retrieve_index + base + child).to(tl.int64)
                    tl.store(accept_index + req * DEPTH + accepted_count, output_slot)
                    row = child
                    coin = tl.load(coins + base + child).to(tl.float32)
                else:
                    rejected = token
                    active = 0
    tl.store(accept_token_num + req, accepted_count)
    tl.store(metadata + req * 2, row)
    tl.store(metadata + req * 2 + 1, output_slot)
    tl.store(rejected_token + req, rejected)


@triton.jit
def _selected_restore_kernel(
    probs,
    keys,
    denominator,
    metadata,
    rejected,
    output,
    B: tl.constexpr,
    D: tl.constexpr,
    V: tl.constexpr,
    NB: tl.constexpr,
    BLOCK: tl.constexpr,
):
    for flat in tl.range(tl.program_id(0), B * NB, tl.num_programs(0)):
        req = (flat // NB).to(tl.int64)
        offsets = (flat % NB) * BLOCK + tl.arange(0, BLOCK)
        valid = offsets < V
        row = req * D + tl.load(metadata + req * 2).to(tl.int64)
        key = tl.load(keys + req * V + offsets, valid, other=0.0)
        value = tl.load(probs + row * V + offsets, valid, other=0.0)
        value = tl.where(key == offsets.to(tl.float32) * 2.0, 0.0, value)
        value = tl.div_rn(value, tl.load(denominator + row))
        # Single-child rejection sets exactly one residual entry to zero.
        # Equivalent to max(p - p, 0); no full B*D*V scratch tensor is needed.
        value = tl.where(offsets == tl.load(rejected + req), 0.0, value)
        tl.store(output + req * V + offsets, value, valid)


def _prepare_selected(probs, top_ps, candidates, next_token, sorted_inputs=None):
    """Internal preparation; shared sort inputs allow exact tie-rule tests."""
    rows, vocab = probs.shape
    values, indices = (
        probs.sort(dim=-1, descending=True) if sorted_inputs is None else sorted_inputs
    )
    cumulative = values.cumsum(-1)
    thresholds = _as_batch_threshold(top_ps, probs, probs.dtype, "top_ps").clamp(0, 1)
    blocks = triton.cdiv(vocab, 2048)
    keys = torch.empty_like(values)
    partials = torch.empty((rows, blocks), device=probs.device, dtype=probs.dtype)
    _filter_encode_kernel[(min(rows * blocks, get_device_properties()[1]),)](
        values,
        indices,
        cumulative,
        thresholds,
        keys,
        rows,
        vocab,
        blocks,
        thresholds.stride(0),
        1,
        2048,
        candidates,
        next_token,
        partials,
        DRAFT=candidates.shape[1],
    )
    denominator = values.sum(-1, keepdim=True).clamp_min_(1e-20)
    return keys, partials, denominator


def _finish_selected(
    probs,
    keys,
    partials,
    denominator,
    predicts,
    accept_index,
    accept_token_num,
    candidates,
    retrive_index,
    retrive_next_token,
    uniform_samples,
    uniform_samples_for_final_sampling,
    threshold_single,
    threshold_acc,
):
    batch, draft = candidates.shape
    vocab = probs.shape[-1]
    blocks = triton.cdiv(vocab, 2048)
    metadata = torch.empty((batch, 2), device=probs.device, dtype=torch.int64)
    rejected = torch.empty((batch,), device=probs.device, dtype=torch.int64)
    _selected_accept_kernel[(batch,)](
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        retrive_next_token,
        uniform_samples,
        partials,
        denominator,
        metadata,
        rejected,
        float(threshold_single),
        max(float(threshold_acc), 1e-9),
        draft,
        accept_index.shape[1],
        blocks,
        triton.next_power_of_2(blocks),
    )
    rows = metadata[:, 0] + torch.arange(batch, device=probs.device) * draft
    # This is the only vocabulary-sized gather, and it copies B rows, not B*D.
    # Deliberately use ordinary sort to isolate this experiment from SortV2.
    ordered_keys = keys.index_select(0, rows).sort(dim=-1).values
    residual = torch.empty((batch, vocab), device=probs.device, dtype=probs.dtype)
    programs = min(batch * blocks, get_device_properties()[1])
    _selected_restore_kernel[(programs,)](
        probs,
        ordered_keys,
        denominator,
        metadata,
        rejected,
        residual,
        batch,
        draft,
        vocab,
        blocks,
        2048,
    )
    block_sums = torch.empty((batch, blocks), device=probs.device, dtype=probs.dtype)
    _tree_target_only_block_sum_kernel[(programs,)](
        residual,
        residual,
        metadata,
        block_sums,
        batch,
        draft,
        vocab,
        2048,
        blocks,
        COMPACT_RESIDUAL=True,
    )
    _tree_target_only_sample_kernel[(batch,)](
        predicts,
        residual,
        residual,
        metadata,
        block_sums,
        uniform_samples_for_final_sampling,
        draft,
        vocab,
        2048,
        blocks,
        triton.next_power_of_2(blocks),
        COMPACT_RESIDUAL=True,
    )


def top_p_tree_sampling_selected_row(
    predicts,
    accept_index,
    accept_token_num,
    candidates,
    retrive_index,
    retrive_next_token,
    retrive_next_sibling,
    uniform_samples,
    uniform_samples_for_final_sampling,
    target_probs,
    top_ps,
    threshold_single=1.0,
    threshold_acc=1.0,
    tree_topk=1,
):
    """Mutate the same outputs as tree_speculative_sampling_target_only.

    Requires a single-child tree supplied by the framework, FP32 probabilities,
    valid tree/token indices, and the RN division policy. No device-to-host
    validation or value-dependent Python branch is introduced.
    """
    if tree_topk != 1:
        raise ValueError("selected-row sampling requires tree_topk=1")
    if candidates.ndim != 2 or target_probs.ndim != 2:
        raise ValueError("candidates and flattened target_probs must be 2-D")
    batch, draft = candidates.shape
    if draft < 1 or target_probs.shape[0] != batch * draft:
        raise ValueError("target_probs must have B*D rows, with D >= 1")
    if not 1 <= target_probs.shape[-1] <= 2**23:
        raise ValueError("packed FP32 keys require vocab in [1, 2**23]")
    if target_probs.device.type != "npu" or target_probs.dtype != torch.float32:
        raise ValueError("selected-row sampling requires FP32 NPU probabilities")
    if (
        accept_index.ndim != 2
        or accept_index.shape[0] != batch
        or not 1 <= accept_index.shape[1] <= draft
    ):
        raise ValueError("accept_index must have shape [B, depth], 1 <= depth <= D")
    if (
        predicts.ndim != 1
        or predicts.numel() < batch * draft
        or accept_token_num.shape != (batch,)
    ):
        raise ValueError("invalid predicts or accept_token_num shape")
    if uniform_samples_for_final_sampling.shape != (batch,):
        raise ValueError("final coins must have shape [B]")
    if any(
        t.shape != candidates.shape
        for t in (
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            uniform_samples,
        )
    ):
        raise ValueError("tree tensors and coins must have shape [B, D]")
    if not 0 <= threshold_single <= 1 or not 0 <= threshold_acc <= 1:
        raise ValueError("accept thresholds must be in [0, 1]")
    typed = (
        (predicts, torch.int32),
        (accept_index, torch.int32),
        (accept_token_num, torch.int32),
        (candidates, torch.int64),
        (retrive_index, torch.int64),
        (retrive_next_token, torch.int64),
        (retrive_next_sibling, torch.int64),
        (uniform_samples, torch.float32),
        (uniform_samples_for_final_sampling, torch.float32),
        (target_probs, torch.float32),
    )
    if any(
        t.dtype != dt or t.device != target_probs.device or not t.is_contiguous()
        for t, dt in typed
    ):
        raise ValueError(
            "all tensors must be contiguous, on one NPU, with tree API dtypes"
        )
    if batch == 0:
        return
    keys, partials, denominator = _prepare_selected(
        target_probs, top_ps, candidates, retrive_next_token
    )
    _finish_selected(
        target_probs,
        keys,
        partials,
        denominator,
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        retrive_next_token,
        uniform_samples,
        uniform_samples_for_final_sampling,
        threshold_single,
        threshold_acc,
    )
