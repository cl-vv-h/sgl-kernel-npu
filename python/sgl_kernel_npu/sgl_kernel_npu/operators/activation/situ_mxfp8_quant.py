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
    from ...implementation.torch_native.activation.situ_mxfp8_quant import (
        situ_mxfp8_quant,
    )

    return situ_mxfp8_quant(
        hidden_states, group_list, group_list_type, beta, linear_beta
    )


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
