from __future__ import annotations

import triton
import triton.language as tl

_ELEM_BLOCK_SIZE = 2048


@triton.jit
def _fused_sigmoid_mul_kernel(
    x_ptr,
    gate_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    block_start = pid * BLOCK_SIZE

    for offset in tl.range(block_start, n_elements, num_programs * BLOCK_SIZE):
        idx = offset + tl.arange(0, BLOCK_SIZE)
        mask = idx < n_elements
        x = tl.load(x_ptr + idx, mask=mask).to(tl.float32)
        g = tl.load(gate_ptr + idx, mask=mask).to(tl.float32)
        out = x * tl.sigmoid(g)
        tl.store(out_ptr + idx, out.to(x_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _fused_sigmoid_mul_broadcast_kernel(
    out_ptr,
    gate_ptr,
    x_ptr,
    num_rows,
    hidden_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    for row in tl.range(pid, num_rows, num_programs):
        g = tl.load(gate_ptr + row).to(tl.float32)
        g = tl.sigmoid(g)

        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < hidden_dim
        x = tl.load(x_ptr + row * hidden_dim + offs, mask=mask).to(tl.float32)
        out = x * g
        tl.store(
            out_ptr + row * hidden_dim + offs,
            out.to(x_ptr.dtype.element_ty),
            mask=mask,
        )
