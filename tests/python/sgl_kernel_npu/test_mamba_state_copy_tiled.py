import importlib.util
from pathlib import Path

import pytest
import torch


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("block", [1024, 2048, 4096, 8192])
@pytest.mark.parametrize(
    "batch,strided", [(0, False), (1, False), (5, False), (5, True)]
)
def test_tiled_state_copy(dtype, block, batch, strided, monkeypatch):
    pytest.importorskip("torch_npu")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("requires NPU")
    monkeypatch.setenv("SGL_KERNEL_NPU_STATE_COPY_TILED", "1")
    monkeypatch.setenv("SGL_KERNEL_NPU_STATE_COPY_BLOCK", str(block))
    path = Path(__file__).resolve().parents[3] / "benchmark/bench_mamba_state_copy.py"
    spec = importlib.util.spec_from_file_location("state_copy_bench", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.check_case(batch, 3, 3, 17, 31, 4, dtype, block, strided)
