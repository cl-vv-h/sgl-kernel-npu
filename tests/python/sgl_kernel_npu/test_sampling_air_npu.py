"""Real NPU tests. CPU emulation cannot replace these compilation checks."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("torch_npu")
if not torch.npu.is_available():
    pytest.skip("NPU unavailable", allow_module_level=True)

from sgl_kernel_npu.sample.probability import top_k_renorm_prob, top_p_renorm_prob
from sgl_kernel_npu.sample.probability_air import air_top_p_renorm

path = Path(__file__).resolve().parents[3] / "benchmark/sampling_air_reference.py"
spec = importlib.util.spec_from_file_location("sampling_air_reference", path)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


@pytest.mark.parametrize("block", [256, 512, 1024])
@pytest.mark.parametrize("vocab", [97, 248320])
@pytest.mark.parametrize("scalar", [False, True])
def test_air_cutoff_and_normalization(block, vocab, scalar):
    gen = torch.Generator().manual_seed(17)
    cpu = torch.softmax(torch.randn(5, vocab, generator=gen), -1)
    cpu[0].fill_(1 / vocab)
    cpu[1].zero_()
    cpu[2].zero_()
    cpu[2, -1] = 1
    probs = cpu.npu()
    ps = torch.tensor([0, 0.2, 0.5, 0.95, 1], dtype=torch.float32).npu()
    if scalar:
        ps = ps[3].expand(5)
    actual = air_top_p_renorm(probs, ps, block).cpu().numpy()
    reference.validate(cpu.numpy(), ps.cpu().numpy(), actual)
    torch.testing.assert_close(probs.cpu(), cpu, rtol=0, atol=0)


def test_public_air_after_top_k(monkeypatch):
    monkeypatch.setenv("SGL_KERNEL_NPU_SAMPLING_TRITON", "1")
    monkeypatch.setenv("SGL_KERNEL_NPU_SAMPLING_TRITON_DIV", "rn")
    monkeypatch.setenv("SGL_KERNEL_NPU_SAMPLING_AIR", "1")
    monkeypatch.setenv("SGL_KERNEL_NPU_SAMPLING_AIR_BLOCK", "512")
    gen = torch.Generator().manual_seed(81)
    probs = torch.softmax(torch.randn(4, 2051, generator=gen), -1).npu()
    filtered = top_k_renorm_prob(probs, torch.tensor([1, 7, 511, 2051]).npu())
    actual = top_p_renorm_prob(filtered, 0.95).cpu().numpy()
    reference.validate(filtered.cpu().numpy(), np.full(4, 0.95, np.float32), actual)
