"""Compatibility path for the public HC residual-mixing operator."""

from ..operators.hc.mix_fused import mix_fused


def __getattr__(name):
    from ..implementation.triton.hc import attn_residual as implementation

    return getattr(implementation, name)
