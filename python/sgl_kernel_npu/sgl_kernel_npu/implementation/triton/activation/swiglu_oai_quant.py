"""swiglu_oai + optional quantization for Ascend NPU.

Adapted from: sgl_kernel_npu package (swiglu_quant.py)

Implements swiglu_oai fused with optional per-row int8 quant:

    gate = x1.clamp(-inf, limit)
    up   = x2.clamp(-limit, limit)
    out  = gate * sigmoid(gate * alpha) * (up + 1)

where x = [gate | up] is concatenated along the last dimension.
When quantization is enabled, each row is scaled to int8 with per-row scale.
"""

import triton
import triton.language as tl
import triton.language.extra.cann.extension as al

# ── Tuning / alignment constants ─────────────────────────────────────────
# Per-row output scaled so |out| <= INT8_MAX before the saturating int8 cast.
_INT8_MAX = 127

# Columns per UB tile in the quantization epilogue; tuned for Ascend vector core.
_COL_BLOCK_SIZE = 1536

# NPU requires 64-byte aligned group_list loads; element count depends on dtype.
_UB_ALIGN_BYTES = 64
_INT64_ALIGN_ELEMS = _UB_ALIGN_BYTES // 8  # 8 int64 elems = 64 B
_INT32_ALIGN_ELEMS = _UB_ALIGN_BYTES // 4  # 16 int32 elems = 64 B


@triton.jit
def _swiglu_oai_quant_kernel(
    x_ptr,
    group_list_ptr,
    out_ptr,
    scale_ptr,
    alpha,
    limit,
    total_rows,
    TOTAL_COLS: tl.constexpr,
    HALF_COLS: tl.constexpr,
    COL_BLOCK_SIZE: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    NUM_EXPERTS_ALGIN: tl.constexpr,
    GROUP_LIST_TYPE: tl.constexpr,
    NUM_CORES: tl.constexpr,
    DTYPE_MAX: tl.constexpr,
    SCALE: tl.constexpr,
    HAS_GROUP_LIST: tl.constexpr,
):
    # MoE: derive total_rows from group_list; dense: use the passed-in total_rows.
    if HAS_GROUP_LIST:
        if (
            GROUP_LIST_TYPE == 0
        ):  # cusum (group_list length = NUM_EXPERTS, last element = total)
            total_rows = tl.load(group_list_ptr + NUM_EXPERTS - 1).to(tl.int32)
        else:
            gl_offsets = tl.arange(0, NUM_EXPERTS_ALGIN)
            gl_mask = gl_offsets < NUM_EXPERTS
            group_list = tl.load(group_list_ptr + gl_offsets, gl_mask, other=0).to(
                tl.int32
            )
            total_rows = tl.sum(group_list)

    block_size = (total_rows - 1) // NUM_CORES + 1
    pid = tl.program_id(0)
    row_begin = pid * block_size
    if row_begin >= total_rows:
        return
    row_end = tl.minimum((pid + 1) * block_size, total_rows)

    for row_idx in range(row_begin, row_end):
        # swiglu_oai: gate * sigmoid(gate * alpha) * (up + 1)  with clamping
        x_offsets = row_idx * TOTAL_COLS + tl.arange(0, TOTAL_COLS)
        cur_x = tl.load(x_ptr + x_offsets)
        gate = al.extract_slice(cur_x, offsets=(0,), sizes=(HALF_COLS,), strides=(1,))
        up = al.extract_slice(
            cur_x, offsets=(HALF_COLS,), sizes=(HALF_COLS,), strides=(1,)
        )
        gate = tl.minimum(gate, limit)  # clamp(-inf, limit]
        up = tl.minimum(tl.maximum(up, -limit), limit)  # clamp[-limit, limit]
        out = gate * tl.sigmoid(gate * alpha) * (up + 1.0)

        # quant
        if SCALE:
            scale = tl.max(tl.abs(out)).to(tl.float32) / DTYPE_MAX
            # store scale
            tl.store(scale_ptr + row_idx, scale.to(scale_ptr.dtype.element_ty))
            for col_blk_idx in range(0, HALF_COLS, COL_BLOCK_SIZE):
                tmp_out = al.extract_slice(
                    out, offsets=(col_blk_idx,), sizes=(COL_BLOCK_SIZE,), strides=(1,)
                )
                tmp_out = (tmp_out.to(tl.float32) / scale).to(x_ptr.dtype.element_ty)
                tmp_out = tmp_out.cast(tl.int8, overflow_mode="saturate")

                o_offsets = (
                    row_idx * HALF_COLS + col_blk_idx + tl.arange(0, COL_BLOCK_SIZE)
                )
                mask = (col_blk_idx + tl.arange(0, COL_BLOCK_SIZE)) < HALF_COLS
                tl.store(
                    out_ptr + o_offsets, tmp_out.to(out_ptr.dtype.element_ty), mask=mask
                )
        else:
            # store out
            o_offsets = row_idx * HALF_COLS + tl.arange(0, HALF_COLS)
            tl.store(out_ptr + o_offsets, out.to(out_ptr.dtype.element_ty))
