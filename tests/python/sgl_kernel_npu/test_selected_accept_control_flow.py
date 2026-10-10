"""CPU semantic execution of the actual accept bodies, not Triton compilation.

The small NumPy adapter exercises masked accesses and state transitions against
our original full-tree accept kernel. Device lowering is tested separately.
"""

import ast
from pathlib import Path

import numpy as np
import pytest


class Tensor(np.ndarray):
    def to(self, dtype):
        return self.astype(dtype).view(Tensor)


def tensor(value, dtype=None):
    return np.asarray(value, dtype=dtype).view(Tensor)


class Pointer:
    def __init__(self, values, offset=0):
        self.values = values.reshape(-1)
        self.offset = np.asarray(offset)

    def __add__(self, offset):
        return Pointer(self.values, self.offset + offset)


class Language:
    int64 = np.int64
    int32 = np.int32
    int1 = np.bool_
    float32 = np.float32
    static_range = staticmethod(range)

    def __init__(self, program):
        self.program = program

    def program_id(self, axis):
        return tensor(self.program, np.int32)

    def full(self, shape, value, dtype):
        return tensor(np.full(shape, value, dtype=dtype))

    def arange(self, start, end):
        return tensor(np.arange(start, end, dtype=np.int32))

    def load(self, ptr, mask=True, other=0):
        offsets, mask = np.broadcast_arrays(ptr.offset, mask)
        # Masked pointers may be out of range; they must never be dereferenced.
        safe = np.where(mask, offsets, 0)
        return tensor(np.where(mask, ptr.values[safe], other), ptr.values.dtype)

    def store(self, ptr, value, mask=True):
        offsets, values, masks = np.broadcast_arrays(ptr.offset, value, mask)
        ptr.values[offsets[masks]] = values[masks]

    def max(self, value, axis):
        return tensor(np.max(value, axis=axis))

    def where(self, condition, left, right):
        return tensor(np.where(condition, left, right))

    def div_rn(self, numerator, denominator):
        return tensor(np.divide(numerator, denominator, dtype=np.float32))


def load_body(filename, function, language):
    root = (
        Path(__file__).resolve().parents[3]
        / "python/sgl_kernel_npu/sgl_kernel_npu/sample"
    )
    tree = ast.parse((root / filename).read_text())
    fn = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function
    )
    fn.decorator_list = []
    for arg in fn.args.args:
        arg.annotation = None
    module = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"tl": language}
    exec(compile(module, str(root / filename), "exec"), namespace)
    return namespace[function]


@pytest.mark.parametrize(
    "draft,depth,blocks", [(1, 1, 1), (4, 4, 122), (4, 3, 76), (8, 8, 1)]
)
@pytest.mark.parametrize("single,acc", [(1.0, 1.0), (0.5, 0.7), (0.0, 1e-9)])
def test_masked_accept_matches_original(draft, depth, blocks, single, acc):
    rng = np.random.default_rng(71)
    batch, vocab = 8, 17
    candidates = rng.integers(vocab, size=(batch, draft), dtype=np.int64)
    index = rng.permutation(batch * draft).reshape(batch, draft).astype(np.int64)
    children = np.full((batch, draft), -1, dtype=np.int64)
    siblings = np.full_like(children, -1)
    for req in range(batch):
        order = [0] + rng.permutation(np.arange(1, draft)).tolist()
        for parent, child in zip(order, order[1:]):
            children[req, parent] = child
        if req % 3 == 0:
            children[req, order[draft // 2]] = -1
    coins = rng.random((batch, draft), dtype=np.float32)
    coins[0] = 0
    coins[1] = 1
    partials = np.zeros((batch * draft, blocks), dtype=np.float32)
    denominator = rng.uniform(0.1, 1, batch * draft).astype(np.float32)
    probs = np.zeros((batch, draft, vocab), dtype=np.float32)
    for req in range(batch):
        for row in range(draft):
            child = children[req, row]
            if child >= 0:
                numerator = np.float32(rng.uniform(0, denominator[req * draft + row]))
                if req == 2:
                    numerator = np.float32(0)
                if req == 3:
                    numerator = denominator[req * draft + row]
                partials[req * draft + row, rng.integers(blocks)] = numerator
                probs[req, row, candidates[req, child]] = np.float32(
                    numerator / denominator[req * draft + row]
                )
    scratch = np.zeros_like(probs)
    expected = [
        np.zeros(batch * draft, np.int32),
        np.full((batch, depth), -1, np.int32),
        np.empty(batch, np.int32),
        np.empty((batch, 2), np.int64),
    ]
    actual = [x.copy() for x in expected]
    rejected = np.empty(batch, np.int64)
    for req in range(batch):
        tl = Language(req)
        old = load_body(
            "tree_speculative_sampling_target_only.py",
            "_tree_target_only_accept_kernel",
            tl,
        )
        old(
            *map(
                Pointer,
                [
                    *expected[:3],
                    candidates,
                    index,
                    children,
                    siblings,
                    coins,
                    probs,
                    scratch,
                    expected[3],
                ],
            ),
            np.float32(single),
            np.float32(acc),
            draft,
            depth,
            vocab
        )
        new = load_body("selected_row.py", "_selected_accept_kernel", tl)
        new(
            *map(
                Pointer,
                [
                    *actual[:3],
                    candidates,
                    index,
                    children,
                    coins,
                    partials,
                    denominator,
                    actual[3],
                    rejected,
                ],
            ),
            np.float32(single),
            np.float32(acc),
            draft,
            depth,
            blocks,
            1 << (blocks - 1).bit_length()
        )
    for a, e in zip(actual, expected):
        np.testing.assert_array_equal(a, e)
    for req in range(batch):
        row = actual[3][req, 0]
        residual = probs[req, row].copy()
        if rejected[req] >= 0:
            residual[rejected[req]] = 0
        np.testing.assert_array_equal(
            residual, np.maximum(probs[req, row] - scratch[req, row], 0)
        )
