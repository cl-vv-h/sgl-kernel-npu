"""Public situ_mxfp8_quant operator with explicit implementation selection."""

import torch

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl(
    "activation.situ_mxfp8_quant", "auto", supported_generations=[Generation.ARCH35]
)
@register_impl(
    "activation.situ_mxfp8_quant", "ascendc", supported_generations=[Generation.ARCH35]
)
def _ascendc_impl(
    hidden_states: torch.Tensor,
    group_list: torch.Tensor,
    group_list_type: int = 1,
    beta: float = 4.0,
    linear_beta: float = 25.0,
):
    """Fuse Kimi-K3 SiTU and per-32-element MXFP8 quantization.

    ``hidden_states`` is the BF16 GMM1 output with shape ``[capacity, 6144]``.
    Only rows selected by ``group_list`` are processed. Returned scales have
    the same logical layout as ``npu_dynamic_mx_quant``: ``[capacity, 48, 2]``.
    """
    if hidden_states.ndim != 2 or hidden_states.shape[1] != 6144:
        raise ValueError(
            "situ_mxfp8_quant expects hidden_states shape [capacity, 6144], "
            f"got {tuple(hidden_states.shape)}"
        )
    if group_list_type not in (0, 1):
        raise ValueError(f"group_list_type must be 0 or 1, got {group_list_type}")
    return torch.ops.npu.situ_mxfp8_quant(
        hidden_states,
        group_list,
        group_list_type,
        float(beta),
        float(linear_beta),
    )


@register_impl("activation.situ_mxfp8_quant", "torch_native")
def _torch_native_impl(
    hidden_states, group_list, group_list_type=1, beta=4.0, linear_beta=25.0
):
    import torch

    from ...implementation.torch_native.activation.situ_mxfp8_quant import (
        situ_mxfp8_quant,
    )

    if hidden_states.ndim != 2 or hidden_states.shape[1] != 6144:
        raise ValueError(
            "situ_mxfp8_quant expects hidden_states shape [capacity, 6144]"
        )
    if hidden_states.dtype != torch.bfloat16 or not hidden_states.is_contiguous():
        raise ValueError("hidden_states must be contiguous BF16")
    if hidden_states.shape[0] == 0 or group_list.ndim != 1 or group_list.numel() == 0:
        raise ValueError("capacity and group_list must be nonempty")
    if group_list.dtype not in (torch.int32, torch.int64):
        raise ValueError("group_list must use int32 or int64")
    if group_list_type not in (0, 1) or beta <= 0 or linear_beta <= 0:
        raise ValueError("invalid group_list_type or SiTU coefficients")
    capacity = hidden_states.shape[0]
    count = group_list[-1] if group_list_type == 0 else group_list.sum()
    valid = max(0, min(capacity, int(count.item())))
    payload = torch.empty(
        (capacity, 3072), dtype=torch.float8_e4m3fn, device=hidden_states.device
    )
    scales = torch.empty(
        (capacity, 48, 2), dtype=torch.float8_e8m0fnu, device=hidden_states.device
    )

    return situ_mxfp8_quant(hidden_states, valid, beta, linear_beta, payload, scales)


@register_operator("activation.situ_mxfp8_quant")
def situ_mxfp8_quant(
    hidden_states,
    group_list,
    group_list_type=1,
    beta=4.0,
    linear_beta=25.0,
    *,
    impl="auto",
):
    """Dispatch to ascendc by default; use impl="torch_native" for the reference.

    Preserve the existing input, output, and in-place behavior.
    """
    return get_impl_by_name("activation.situ_mxfp8_quant", impl)(
        hidden_states, group_list, group_list_type, beta, linear_beta
    )
