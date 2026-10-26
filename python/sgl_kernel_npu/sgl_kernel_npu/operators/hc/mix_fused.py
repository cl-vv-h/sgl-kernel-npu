"""Attention residual RMS scoring, softmax and weighted mixing."""

import torch

from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl("hc.mix_fused", "auto")
@register_impl("hc.mix_fused", "triton")
def _triton_impl(
    prefix_sum: torch.Tensor,
    bank: torch.Tensor,
    num_valid_blocks: int,
    combined_weight: torch.Tensor,
    variance_epsilon: float,
) -> torch.Tensor:
    """Ascend Kimi-K3 attention-residual score and combine pipeline.

    Keep scoring, softmax, and mixing in one persistent vector-core kernel to
    avoid materializing the score matrix and launching a second kernel.  N and
    B stay dynamic so prefill/decode shapes reuse the same compilation.
    """
    import triton

    from ...implementation.triton.hc.attn_residual import _mix_fused_kernel
    from ...utils.triton_utils import get_device_properties

    num_tokens, hidden_size = prefix_sum.shape
    if num_tokens == 0:
        return prefix_sum
    if not 0 <= num_valid_blocks <= bank.shape[1]:
        raise ValueError("num_valid_blocks must fit within the residual bank")

    output = torch.empty_like(prefix_sum)
    _, num_vector_cores = get_device_properties()
    _mix_fused_kernel[(num_vector_cores,)](
        prefix_sum,
        bank,
        combined_weight,
        output,
        num_tokens,
        num_valid_blocks,
        prefix_sum.stride(0),
        bank.stride(0),
        bank.stride(1),
        output.stride(0),
        H=hidden_size,
        EPS=variance_epsilon,
        NUM_CORES=num_vector_cores,
        NB=triton.next_power_of_2(num_valid_blocks + 1),
        multibuffer=True,
    )
    return output


@register_impl("hc.mix_fused", "torch_native")
def _torch_native_impl(
    prefix_sum, bank, num_valid_blocks, combined_weight, variance_epsilon
):
    from ...implementation.torch_native.hc.mix_fused import mix_fused

    num_tokens, hidden_size = prefix_sum.shape
    if num_tokens == 0:
        return prefix_sum
    if not 0 <= num_valid_blocks <= bank.shape[1]:
        raise ValueError("num_valid_blocks must fit within the residual bank")
    if bank.ndim != 3 or bank.shape[0] != num_tokens or bank.shape[2] != hidden_size:
        raise ValueError("bank must have shape [tokens, blocks, hidden]")
    if combined_weight.numel() != hidden_size:
        raise ValueError("combined_weight must contain hidden elements")
    candidates = torch.cat((bank[:, :num_valid_blocks], prefix_sum.unsqueeze(1)), dim=1)
    return mix_fused(
        candidates, combined_weight.reshape(-1), variance_epsilon, prefix_sum.dtype
    )


@register_operator("hc.mix_fused")
def mix_fused(
    prefix_sum,
    bank,
    num_valid_blocks,
    combined_weight,
    variance_epsilon,
    *,
    impl="auto"
):
    """Mix valid residual-bank blocks with prefix_sum; auto uses Triton."""
    return get_impl_by_name("hc.mix_fused", impl)(
        prefix_sum, bank, num_valid_blocks, combined_weight, variance_epsilon
    )
