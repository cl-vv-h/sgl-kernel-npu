"""Exact fixed-coin regression against full V1+RN target-only tree sampling."""

import importlib.util
from pathlib import Path

import pytest
import torch


@pytest.mark.parametrize(
    "vocab,draft,depth,short", [(1, 1, 1, False), (97, 4, 4, False), (2051, 4, 3, True)]
)
@pytest.mark.parametrize("case", ["random", "ties", "zeros", "onehot"])
@pytest.mark.parametrize("single,acc", [(1.0, 1.0), (0.0, 0.0), (0.5, 0.7)])
def test_exact_selected_row(vocab, draft, depth, short, case, single, acc):
    pytest.importorskip("torch_npu")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("requires NPU")
    path = (
        Path(__file__).resolve().parents[3] / "benchmark/bench_sampling_selected_row.py"
    )
    spec = importlib.util.spec_from_file_location("selected_bench", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.check_shared_sort(vocab, draft, case, short, single, acc, depth)
