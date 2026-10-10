"""Execute the AIR Triton bodies through a CPU adapter to check algorithm logic.

This does not validate Ascend lowering or its floating-point reduction order.
The independent reference sorts in FP64 and retains every cutoff tie.
"""

import ast
import importlib.util
from pathlib import Path

import numpy as np
import pytest


class Tensor(np.ndarray):
    def to(self, dtype, bitcast=False):
        return (self.view(dtype) if bitcast else self.astype(dtype)).view(Tensor)


def tensor(value, dtype=None):
    return np.asarray(value, dtype=dtype).view(Tensor)


class Pointer:
    def __init__(self, values, offset=0):
        self.values, self.offset = values.reshape(-1), np.asarray(offset)

    def __add__(self, value):
        return Pointer(self.values, self.offset + value)


class Language:
    int64, int32, uint32, float32 = np.int64, np.int32, np.uint32, np.float32

    @staticmethod
    def range(start, end, step=1):
        for value in range(int(start), int(end), int(step)):
            yield tensor(value, np.int32)

    def __init__(self):
        self.program, self.programs = 0, 1

    def program_id(self, axis):
        return tensor(self.program, np.int32)

    def num_programs(self, axis):
        return self.programs

    def full(self, shape, value, dtype):
        return tensor(np.full(shape, value, dtype=dtype))

    def arange(self, start, end):
        return tensor(np.arange(start, end, dtype=np.int32))

    def load(self, ptr, mask=True, other=0):
        offsets, mask = np.broadcast_arrays(ptr.offset, mask)
        return tensor(
            np.where(mask, ptr.values[np.where(mask, offsets, 0)], other),
            ptr.values.dtype,
        )

    def store(self, ptr, value, mask=True):
        offsets, values, masks = np.broadcast_arrays(ptr.offset, value, mask)
        ptr.values[offsets[masks]] = values[masks]

    def sum(self, value, axis):
        return tensor(np.sum(value, axis=axis))

    def min(self, value, axis):
        return tensor(np.min(value, axis=axis))

    def max(self, value, axis):
        return tensor(np.max(value, axis=axis))

    def cumsum(self, value, axis):
        return tensor(np.cumsum(value, axis=axis))

    def where(self, condition, left, right):
        return tensor(np.where(condition, left, right))

    def maximum(self, left, right):
        return tensor(np.maximum(left, right))

    def div_rn(self, left, right):
        return tensor(np.divide(left, right, dtype=np.float32))


def load_kernels(tl):
    path = (
        Path(__file__).resolve().parents[3]
        / "python/sgl_kernel_npu/sgl_kernel_npu/sample/probability_air.py"
    )
    tree = ast.parse(path.read_text())
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.endswith("_kernel"):
            node.decorator_list = []
            for arg in node.args.args:
                arg.annotation = None
            nodes.append(node)
    tree = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(tree)
    scope = {"tl": tl}
    exec(compile(tree, str(path), "exec"), scope)
    return scope


def emulate(probs, top_ps, block=256, scalar=False):
    rows, vocab = probs.shape
    tiles = (vocab + block - 1) // block
    pad = 1 << (tiles - 1).bit_length()
    prefix = np.empty(rows, np.int32)
    remaining = np.empty(rows, np.float32)
    hist = np.empty((rows, tiles, 16), np.float32)
    partial = np.empty((rows, tiles), np.float32)
    denom = np.empty(rows, np.float32)
    out = np.empty_like(probs)
    tl = Language()
    kernels = load_kernels(tl)

    def launch(name, programs, *args):
        tl.programs = programs
        for tl.program in range(programs):
            kernels[name](
                *[Pointer(a) if isinstance(a, np.ndarray) else a for a in args]
            )

    grid, stride = min(rows * tiles, 3), 0 if scalar else 1
    for shift in range(28, -1, -4):
        launch(
            "_air_histogram_kernel",
            grid,
            probs,
            prefix,
            hist,
            rows,
            vocab,
            tiles,
            shift,
            shift == 28,
            block,
        )
        launch(
            "_air_select_kernel",
            rows,
            hist,
            prefix,
            remaining,
            top_ps,
            tiles,
            pad,
            stride,
            shift,
            shift == 28,
        )
    launch(
        "_air_kept_mass_kernel",
        grid,
        probs,
        prefix,
        top_ps,
        partial,
        rows,
        vocab,
        tiles,
        stride,
        block,
    )
    launch("_air_denominator_kernel", rows, partial, denom, tiles, pad)
    launch(
        "_air_apply_kernel",
        grid,
        probs,
        prefix,
        top_ps,
        denom,
        out,
        rows,
        vocab,
        tiles,
        stride,
        block,
    )
    return out


def oracle(probs, ps):
    out = np.zeros_like(probs)
    for row, p in enumerate(np.broadcast_to(ps, (len(probs),))):
        values = np.sort(probs[row].astype(np.float64))[::-1]
        if not values.sum():
            continue
        if p >= 1:
            kept = probs[row].astype(np.float64)
        else:
            idx = min(
                np.searchsorted(np.cumsum(values), p * values.sum(), side="left"),
                len(values) - 1,
            )
            cutoff = values[idx]
            kept = np.where(probs[row] >= cutoff, probs[row], 0).astype(np.float64)
        out[row] = kept / max(kept.sum(), 1e-20)
    return out


@pytest.mark.parametrize("vocab", [1, 4, 97, 2051])
@pytest.mark.parametrize(
    "case", ["random", "ties", "uniform", "zeros", "onehot", "tiny", "unnormalized"]
)
@pytest.mark.parametrize("scalar", [False, True])
def test_air_bodies_match_sort_oracle(vocab, case, scalar):
    rng = np.random.default_rng(17)
    p = rng.uniform(0.01, 1, (5, vocab)).astype(np.float32)
    if case == "ties":
        p = np.floor(p * 4)
    elif case == "uniform":
        p.fill(1)
    elif case == "zeros":
        p.fill(0)
    elif case == "onehot":
        p.fill(0)
        p[:, -1] = 1
    p = p / np.maximum(p.sum(-1, keepdims=True), 1e-20)
    if case == "tiny":
        p *= np.float32(1e-25)
    if case == "unnormalized":
        p *= np.array([0.03, 0.3, 2.0, 13.0, 100.0], np.float32)[:, None]
    ps = np.array([0.95] if scalar else [0, 0.2, 0.5, 0.95, 1], np.float32)
    actual, expected = emulate(p, ps, scalar=scalar), oracle(p, ps)
    np.testing.assert_array_equal(actual > 0, expected > 0)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-8)


@pytest.mark.parametrize("block", [256, 512, 1024])
def test_public_size_and_permutations(block):
    rng = np.random.default_rng(81)
    probs = rng.uniform(0, 1, (2, 248320)).astype(np.float32)
    probs /= probs.sum(-1, keepdims=True)
    ps = np.array([0.95, 0.95], np.float32)
    actual, expected = emulate(probs, ps, block), oracle(probs, ps)
    path = Path(__file__).resolve().parents[3] / "benchmark/sampling_air_reference.py"
    spec = importlib.util.spec_from_file_location("sampling_air_reference", path)
    reference_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference_module)
    # The FP32 radix reduction can shift one boundary token relative to FP64.
    # Check complete threshold sets, bracketing mass, normalization and TV;
    # small exact edge cases above still require identical support.
    reference_module.validate(probs, ps, actual)


def test_exact_cutoff_ties_are_kept():
    p = np.array([[0.5, 0.25, 0.25, 0], [0.25, 0.25, 0.25, 0.25]], np.float32)
    actual = emulate(p, np.array([0.5, 0.5], np.float32))
    np.testing.assert_array_equal(actual[0], [1, 0, 0, 0])
    np.testing.assert_array_equal(actual[1], [0.25, 0.25, 0.25, 0.25])
