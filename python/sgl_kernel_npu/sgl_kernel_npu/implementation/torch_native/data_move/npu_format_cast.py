"""Base-format reference, with explicit rejection of hardware-private layouts."""

import torch


def npu_format_cast(input, acl_format, *, customize_dtype=None, input_dtype=None):
    if (
        acl_format not in (0, 2)
        or customize_dtype is not None
        or input_dtype is not None
    ):
        raise NotImplementedError(
            "torch_native cannot construct NPU private storage descriptors or custom packed dtypes; use torch_npu"
        )
    if acl_format == 0 and input.ndim != 4:
        raise ValueError("NCHW format requires a rank-4 tensor")
    # Allocate a base dense tensor rather than cloning private source storage.
    return torch.empty(input.shape, dtype=input.dtype, device=input.device).copy_(input)
