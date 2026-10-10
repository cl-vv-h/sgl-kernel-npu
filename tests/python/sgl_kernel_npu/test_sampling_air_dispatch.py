"""Metadata-only dispatch tests, runnable without an NPU installation."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


@pytest.fixture
def probability(monkeypatch):
    path = (
        Path(__file__).resolve().parents[3]
        / "python/sgl_kernel_npu/sgl_kernel_npu/sample/probability.py"
    )
    spec = importlib.util.spec_from_file_location(
        "sgl_kernel_npu.sample.probability", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.delenv("SGL_KERNEL_NPU_SAMPLING_AIR", raising=False)
    monkeypatch.delenv("SGL_KERNEL_NPU_SAMPLING_AIR_BLOCK", raising=False)
    return module


def test_air_dispatch_precedes_sort(probability, monkeypatch):
    calls = []
    fake = types.ModuleType("sgl_kernel_npu.sample.probability_air")
    fake.air_top_p_renorm = lambda *args: calls.append(args) or args[0]
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    monkeypatch.setattr(probability, "_use_triton", lambda _: True)
    monkeypatch.setenv("SGL_KERNEL_NPU_SAMPLING_AIR", "1")
    monkeypatch.setenv("SGL_KERNEL_NPU_SAMPLING_AIR_BLOCK", "256")

    def forbidden(*args, **kwargs):
        pytest.fail("AIR dispatch must not execute sort")

    monkeypatch.setattr(torch.Tensor, "sort", forbidden)
    probs = torch.ones(3, 17) / 17
    assert probability.top_p_renorm_prob(probs, 1.5) is probs
    assert calls[0][2] == 256
    torch.testing.assert_close(calls[0][1], torch.ones(3))


@pytest.mark.parametrize("enabled", [False, True])
def test_default_and_unsupported_shape_use_v1(probability, monkeypatch, enabled):
    fake = types.ModuleType("sgl_kernel_npu.sample.probability_triton")
    calls = []
    fake.filter_and_renorm = (
        lambda *args, **kwargs: calls.append((args, kwargs)) or args[0]
    )
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    monkeypatch.setattr(probability, "_use_triton", lambda _: True)
    monkeypatch.setenv("SGL_KERNEL_NPU_SAMPLING_TRITON_DIV", "rn")
    if enabled:
        monkeypatch.setenv("SGL_KERNEL_NPU_SAMPLING_AIR", "1")
        monkeypatch.setenv("SGL_KERNEL_NPU_SAMPLING_AIR_BLOCK", "256")
    probs = torch.ones(1, 262145 if enabled else 17)
    assert probability.top_p_renorm_prob(probs, 0.95) is probs
    assert len(calls) == 1 and calls[0][1]["division"] == "rn"


def test_cpu_ignores_air_flag(probability, monkeypatch):
    probs = torch.tensor([[0.5, 0.25, 0.25]])
    expected = probability.top_p_renorm_prob(probs, 0.95)
    monkeypatch.setenv("SGL_KERNEL_NPU_SAMPLING_AIR", "1")
    torch.testing.assert_close(probability.top_p_renorm_prob(probs, 0.95), expected)


def test_invalid_block_rejected(probability, monkeypatch):
    monkeypatch.setattr(probability, "_use_triton", lambda _: True)
    monkeypatch.setenv("SGL_KERNEL_NPU_SAMPLING_AIR", "1")
    monkeypatch.setenv("SGL_KERNEL_NPU_SAMPLING_AIR_BLOCK", "128")
    with pytest.raises(ValueError, match="256, 512 or 1024"):
        probability.top_p_renorm_prob(torch.ones(1, 4), 0.95)
