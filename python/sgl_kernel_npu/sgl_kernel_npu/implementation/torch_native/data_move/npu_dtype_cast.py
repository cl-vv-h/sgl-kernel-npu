import torch


def npu_dtype_cast(input, dtype):
    if not isinstance(dtype, torch.dtype) or "float4" in str(dtype):
        raise NotImplementedError(
            "torch_native dtype_cast requires a standard unpacked torch.dtype; use torch_npu for packed FP4"
        )
    return input.to(dtype=dtype)
