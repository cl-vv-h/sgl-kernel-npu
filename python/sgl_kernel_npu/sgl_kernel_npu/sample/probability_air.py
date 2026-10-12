# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 by FlashInfer team.
# Modifications: NPU Triton implementation using partitioned reductions,
# four-bit radix passes, bounded candidate compaction and RN normalization.
# Reference: flashinfer/include/flashinfer/air_top_p.cuh at
# c94c1a9e4bf151e56080168ac880ebf0651af317. License and attribution below.
"""Experimental AIR-style top-p renormalization for FP32 NPU probabilities.

This is a weighted radix SELECT, not a radix sort. Each pass refines only the
selected bit prefix. After three passes, compact the boundary candidates into
bounded per-tile storage. Dense tiles fall back to the original input; candidate
overflow never truncates the distribution. No global atomics are required.

Inputs must be finite, nonnegative probabilities. All ties at the cutoff are
kept. The target mass is top_p * row_sum; this is not bitwise V1 semantics.
"""

import os

import torch
import triton
import triton.language as tl
from sgl_kernel_npu.utils.triton_utils import get_device_properties

_RADIX_BITS = 4
_BINS = 1 << _RADIX_BITS


# Batch changes only the runtime loop bound, not the tile layout. Avoid compiling
# another set of radix kernels whenever the active request count changes.
@triton.jit(do_not_specialize=["ROWS"])
def _air_histogram_kernel(
    probs,
    prefix,
    histogram,
    ROWS,
    VOCAB: tl.constexpr,
    TILES: tl.constexpr,
    SHIFT: tl.constexpr,
    FIRST: tl.constexpr,
    BLOCK: tl.constexpr,
):
    bins = tl.arange(0, 16)
    # Only bucket labels use FP16: integers 0..15 are exactly representable.
    # Ascend materializes both comparison operands when broadcasting them.
    compare_bins = bins.to(tl.float16)
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
        bucket = ((key >> SHIFT) & 15).to(tl.float16)
        # Small, dense local histogram. Do not substitute a global atomic_add
        # per token: concentrated probability buckets would serialize it.
        # Mask in one dimension before broadcasting. This avoids materializing
        # and combining a separate [16, BLOCK] validity mask on Ascend. Values
        # and reductions remain FP32, with the same reduction axis and order.
        masked_value = tl.where(valid, value, 0.0)
        mass = tl.sum(
            tl.where(
                compare_bins[:, None] == bucket[None, :], masked_value[None, :], 0.0
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


@triton.jit(do_not_specialize=["ROWS"])
def _air_compact_kernel(
    probs,
    prefix,
    candidates,
    counts,
    ROWS,
    VOCAB: tl.constexpr,
    TILES: tl.constexpr,
    BLOCK: tl.constexpr,
    CAPACITY: tl.constexpr,
):
    # The first three radix passes have resolved bits [31:20]. Each tile owns
    # its output slice. The exclusive ranks of matching lanes are unique.
    for flat in tl.range(tl.program_id(0), ROWS * TILES, tl.num_programs(0)):
        row = (flat // TILES).to(tl.int64)
        offset = (flat % TILES) * BLOCK + tl.arange(0, BLOCK)
        value = tl.load(probs + row * VOCAB + offset, offset < VOCAB, other=0.0)
        key = value.to(tl.uint32, bitcast=True) ^ 0x7FFFFFFF
        previous = tl.load(prefix + row).to(tl.uint32)
        match = (offset < VOCAB) & (value > 0.0) & ((key >> 20) == (previous >> 20))
        count = tl.sum(match.to(tl.int32), 0)
        tl.store(counts + flat, count)
        if (count > 0) & (count <= CAPACITY):
            rank = tl.cumsum(match.to(tl.int32), 0) - 1
            tl.store(
                candidates + flat.to(tl.int64) * CAPACITY + rank,
                value,
                match,
            )


@triton.jit
def _air_boundary_histogram(value, valid, previous, SHIFT: tl.constexpr):
    bins = tl.arange(0, 16).to(tl.float16)
    key = value.to(tl.uint32, bitcast=True) ^ 0x7FFFFFFF
    valid = valid & ((key >> (SHIFT + 4)) == (previous >> (SHIFT + 4)))
    bucket = ((key >> SHIFT) & 15).to(tl.float16)
    masked_value = tl.where(valid, value, 0.0)
    return tl.sum(
        tl.where(
            bins[:, None] == bucket[None, :], masked_value[None, :], 0.0
        ),
        1,
    )


@triton.jit(do_not_specialize=["ROWS"])
def _air_compact_histogram_kernel(
    probs,
    prefix,
    candidates,
    counts,
    histogram,
    ROWS,
    VOCAB: tl.constexpr,
    TILES: tl.constexpr,
    SHIFT: tl.constexpr,
    BLOCK: tl.constexpr,
    CAPACITY: tl.constexpr,
):
    bins = tl.arange(0, 16)
    for flat in tl.range(tl.program_id(0), ROWS * TILES, tl.num_programs(0)):
        row = (flat // TILES).to(tl.int64)
        count = tl.load(counts + flat)
        previous = tl.load(prefix + row).to(tl.uint32)
        mass = tl.full((16,), 0.0, tl.float32)
        if count > CAPACITY:
            # Uniform/tied or otherwise dense tiles retain the old full scan.
            # Never read the uninitialized compact buffer for these tiles.
            offset = (flat % TILES) * BLOCK + tl.arange(0, BLOCK)
            valid = offset < VOCAB
            value = tl.load(probs + row * VOCAB + offset, valid, other=0.0)
            mass = _air_boundary_histogram(value, valid, previous, SHIFT)
        elif count > 0:
            lane = tl.arange(0, CAPACITY)
            valid = lane < count
            value = tl.load(
                candidates + flat.to(tl.int64) * CAPACITY + lane, valid, other=0.0
            )
            mass = _air_boundary_histogram(value, valid, previous, SHIFT)
        # Empty tiles must overwrite all bins on every pass, not reuse the
        # previous histogram. All-zero rows still follow _air_select_kernel.
        tl.store(histogram + flat.to(tl.int64) * 16 + bins, mass)


@triton.jit(do_not_specialize=["ROWS"])
def _air_kept_mass_kernel(
    probs,
    prefix,
    top_ps,
    partial_sums,
    ROWS,
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


@triton.jit(do_not_specialize=["ROWS"])
def _air_apply_kernel(
    probs,
    prefix,
    top_ps,
    denominator,
    output,
    ROWS,
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


def air_top_p_renorm(probs, top_ps, block_size=512, *, compact=None):
    """Return a full [rows, vocab] tensor in vocabulary order, using RN division.

    top_ps is a clamped FP32 vector [rows] (stride zero is supported). No tensor
    values are inspected on the host. Selection retains every cutoff tie;
    output is not guaranteed bitwise identical to the sorted V1 path.
    Compact reductions can also round differently from the full-scan AIR path.
    Set compact=False or SGL_KERNEL_NPU_SAMPLING_AIR_COMPACT=0 for that baseline.
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
    if compact is None:
        compact = os.environ.get("SGL_KERNEL_NPU_SAMPLING_AIR_COMPACT", "1") == "1"
    capacity = 64
    if compact:
        candidates = torch.empty(
            (rows, tiles, capacity), device=probs.device, dtype=torch.float32
        )
        counts = torch.empty((rows, tiles), device=probs.device, dtype=torch.int32)

    for shift in range(28, -1, -4):
        if compact and shift < 20:
            _air_compact_histogram_kernel[grid](
                probs,
                prefix,
                candidates,
                counts,
                histogram,
                rows,
                vocab,
                tiles,
                shift,
                block_size,
                capacity,
            )
        else:
            _air_histogram_kernel[grid](
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
        _air_select_kernel[(rows,)](
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
        if compact and shift == 20:
            _air_compact_kernel[grid](
                probs, prefix, candidates, counts, rows, vocab, tiles, block_size, capacity
            )
    _air_kept_mass_kernel[grid](
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
    _air_denominator_kernel[(rows,)](
        partial_sums,
        denominator,
        tiles,
        triton.next_power_of_2(tiles),
    )
    _air_apply_kernel[grid](
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


# The reference's attribution and license are embedded to keep this operator
# self-contained without modifying package metadata or adding ancillary files.
# AIR Top-P algorithm reference and attribution
#
# Copyright (c) 2026 by FlashInfer team.
# Source: https://github.com/flashinfer-ai/flashinfer/blob/c94c1a9e4bf151e56080168ac880ebf0651af317/include/flashinfer/air_top_p.cuh
# License: Apache-2.0 (see FLASHINFER_LICENSE.txt).
# The reference identifies its core algorithm as derived from TensorRT-LLM AIR Top-P (samplingAirTopPKernels.cu).
#
# This port replaces CUDA/CUB, shared-memory atomics and cooperative synchronization with Ascend Triton partitioned reductions. It uses four-bit radix passes, full input rescans instead of candidate compaction, and RN division with a 1e-20 denominator floor. No FlashInfer CUDA binary is bundled.
#
#                                  Apache License
#                            Version 2.0, January 2004
#                         http://www.apache.org/licenses/
#
#    TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION
#
#    1. Definitions.
#
#       "License" shall mean the terms and conditions for use, reproduction,
#       and distribution as defined by Sections 1 through 9 of this document.
#
#       "Licensor" shall mean the copyright owner or entity authorized by
#       the copyright owner that is granting the License.
#
#       "Legal Entity" shall mean the union of the acting entity and all
#       other entities that control, are controlled by, or are under common
#       control with that entity. For the purposes of this definition,
#       "control" means (i) the power, direct or indirect, to cause the
#       direction or management of such entity, whether by contract or
#       otherwise, or (ii) ownership of fifty percent (50%) or more of the
#       outstanding shares, or (iii) beneficial ownership of such entity.
#
#       "You" (or "Your") shall mean an individual or Legal Entity
#       exercising permissions granted by this License.
#
#       "Source" form shall mean the preferred form for making modifications,
#       including but not limited to software source code, documentation
#       source, and configuration files.
#
#       "Object" form shall mean any form resulting from mechanical
#       transformation or translation of a Source form, including but
#       not limited to compiled object code, generated documentation,
#       and conversions to other media types.
#
#       "Work" shall mean the work of authorship, whether in Source or
#       Object form, made available under the License, as indicated by a
#       copyright notice that is included in or attached to the work
#       (an example is provided in the Appendix below).
#
#       "Derivative Works" shall mean any work, whether in Source or Object
#       form, that is based on (or derived from) the Work and for which the
#       editorial revisions, annotations, elaborations, or other modifications
#       represent, as a whole, an original work of authorship. For the purposes
#       of this License, Derivative Works shall not include works that remain
#       separable from, or merely link (or bind by name) to the interfaces of,
#       the Work and Derivative Works thereof.
#
#       "Contribution" shall mean any work of authorship, including
#       the original version of the Work and any modifications or additions
#       to that Work or Derivative Works thereof, that is intentionally
#       submitted to Licensor for inclusion in the Work by the copyright owner
#       or by an individual or Legal Entity authorized to submit on behalf of
#       the copyright owner. For the purposes of this definition, "submitted"
#       means any form of electronic, verbal, or written communication sent
#       to the Licensor or its representatives, including but not limited to
#       communication on electronic mailing lists, source code control systems,
#       and issue tracking systems that are managed by, or on behalf of, the
#       Licensor for the purpose of discussing and improving the Work, but
#       excluding communication that is conspicuously marked or otherwise
#       designated in writing by the copyright owner as "Not a Contribution."
#
#       "Contributor" shall mean Licensor and any individual or Legal Entity
#       on behalf of whom a Contribution has been received by Licensor and
#       subsequently incorporated within the Work.
#
#    2. Grant of Copyright License. Subject to the terms and conditions of
#       this License, each Contributor hereby grants to You a perpetual,
#       worldwide, non-exclusive, no-charge, royalty-free, irrevocable
#       copyright license to reproduce, prepare Derivative Works of,
#       publicly display, publicly perform, sublicense, and distribute the
#       Work and such Derivative Works in Source or Object form.
#
#    3. Grant of Patent License. Subject to the terms and conditions of
#       this License, each Contributor hereby grants to You a perpetual,
#       worldwide, non-exclusive, no-charge, royalty-free, irrevocable
#       (except as stated in this section) patent license to make, have made,
#       use, offer to sell, sell, import, and otherwise transfer the Work,
#       where such license applies only to those patent claims licensable
#       by such Contributor that are necessarily infringed by their
#       Contribution(s) alone or by combination of their Contribution(s)
#       with the Work to which such Contribution(s) was submitted. If You
#       institute patent litigation against any entity (including a
#       cross-claim or counterclaim in a lawsuit) alleging that the Work
#       or a Contribution incorporated within the Work constitutes direct
#       or contributory patent infringement, then any patent licenses
#       granted to You under this License for that Work shall terminate
#       as of the date such litigation is filed.
#
#    4. Redistribution. You may reproduce and distribute copies of the
#       Work or Derivative Works thereof in any medium, with or without
#       modifications, and in Source or Object form, provided that You
#       meet the following conditions:
#
#       (a) You must give any other recipients of the Work or
#           Derivative Works a copy of this License; and
#
#       (b) You must cause any modified files to carry prominent notices
#           stating that You changed the files; and
#
#       (c) You must retain, in the Source form of any Derivative Works
#           that You distribute, all copyright, patent, trademark, and
#           attribution notices from the Source form of the Work,
#           excluding those notices that do not pertain to any part of
#           the Derivative Works; and
#
#       (d) If the Work includes a "NOTICE" text file as part of its
#           distribution, then any Derivative Works that You distribute must
#           include a readable copy of the attribution notices contained
#           within such NOTICE file, excluding those notices that do not
#           pertain to any part of the Derivative Works, in at least one
#           of the following places: within a NOTICE text file distributed
#           as part of the Derivative Works; within the Source form or
#           documentation, if provided along with the Derivative Works; or,
#           within a display generated by the Derivative Works, if and
#           wherever such third-party notices normally appear. The contents
#           of the NOTICE file are for informational purposes only and
#           do not modify the License. You may add Your own attribution
#           notices within Derivative Works that You distribute, alongside
#           or as an addendum to the NOTICE text from the Work, provided
#           that such additional attribution notices cannot be construed
#           as modifying the License.
#
#       You may add Your own copyright statement to Your modifications and
#       may provide additional or different license terms and conditions
#       for use, reproduction, or distribution of Your modifications, or
#       for any such Derivative Works as a whole, provided Your use,
#       reproduction, and distribution of the Work otherwise complies with
#       the conditions stated in this License.
#
#    5. Submission of Contributions. Unless You explicitly state otherwise,
#       any Contribution intentionally submitted for inclusion in the Work
#       by You to the Licensor shall be under the terms and conditions of
#       this License, without any additional terms or conditions.
#       Notwithstanding the above, nothing herein shall supersede or modify
#       the terms of any separate license agreement you may have executed
#       with Licensor regarding such Contributions.
#
#    6. Trademarks. This License does not grant permission to use the trade
#       names, trademarks, service marks, or product names of the Licensor,
#       except as required for reasonable and customary use in describing the
#       origin of the Work and reproducing the content of the NOTICE file.
#
#    7. Disclaimer of Warranty. Unless required by applicable law or
#       agreed to in writing, Licensor provides the Work (and each
#       Contributor provides its Contributions) on an "AS IS" BASIS,
#       WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
#       implied, including, without limitation, any warranties or conditions
#       of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
#       PARTICULAR PURPOSE. You are solely responsible for determining the
#       appropriateness of using or redistributing the Work and assume any
#       risks associated with Your exercise of permissions under this License.
#
#    8. Limitation of Liability. In no event and under no legal theory,
#       whether in tort (including negligence), contract, or otherwise,
#       unless required by applicable law (such as deliberate and grossly
#       negligent acts) or agreed to in writing, shall any Contributor be
#       liable to You for damages, including any direct, indirect, special,
#       incidental, or consequential damages of any character arising as a
#       result of this License or out of the use or inability to use the
#       Work (including but not limited to damages for loss of goodwill,
#       work stoppage, computer failure or malfunction, or any and all
#       other commercial damages or losses), even if such Contributor
#       has been advised of the possibility of such damages.
#
#    9. Accepting Warranty or Additional Liability. While redistributing
#       the Work or Derivative Works thereof, You may choose to offer,
#       and charge a fee for, acceptance of support, warranty, indemnity,
#       or other liability obligations and/or rights consistent with this
#       License. However, in accepting such obligations, You may act only
#       on Your own behalf and on Your sole responsibility, not on behalf
#       of any other Contributor, and only if You agree to indemnify,
#       defend, and hold each Contributor harmless for any liability
#       incurred by, or claims asserted against, such Contributor by reason
#       of your accepting any such warranty or additional liability.
#
#    END OF TERMS AND CONDITIONS
#
#    APPENDIX: How to apply the Apache License to your work.
#
#       To apply the Apache License to your work, attach the following
#       boilerplate notice, with the fields enclosed by brackets "[]"
#       replaced with your own identifying information. (Don't include
#       the brackets!)  The text should be enclosed in the appropriate
#       comment syntax for the file format. We also recommend that a
#       file or class name and description of purpose be included on the
#       same "printed page" as the copyright notice for easier
#       identification within third-party archives.
#
#    Copyright [yyyy] [name of copyright owner]
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
# -------------------------------------------------------------------------------------------------
# Some of the code in this project are adapted from other open-source projects with different
# licenses. This product also bundles some third-party components under other open source licenses.
# This section summarizes those components and their licenses.
# See licenses/ for text of these licenses.
#
# BSD 3-Clause License
# --------------------
#
# include/flashinfer/attention/hopper/epilogue.cuh
# include/flashinfer/attention/hopper/mainloop.cuh
# include/flashinfer/attention/hopper/kernel_traits.cuh
# include/flashinfer/attention/hopper/named_barrier.cuh
# include/flashinfer/attention/hopper/tile_scheduler.cuh
# include/flashinfer/attention/hopper/utils.cuh
#
# BSD 3-Clause "New" License
# --------------------------
#
# 3rdparty/cutlass
# include/flashinfer/attention/hopper/block_sparse_gather.cuh
#
# MIT License
# -----------
#
# 3rdparty/spdlog
# 3rdparty/spdlog/include/spdlog/fmt/bundled (fmt library)
