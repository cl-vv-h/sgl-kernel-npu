"""Compatibility imports for the relocated implementation."""

from ..implementation.triton.mem_cache.fused_sigmoid_mul import fused_sigmoid_mul
from ..operators.mem_cache.fused_sigmoid_mul_broadcast import (
    fused_sigmoid_mul_broadcast,
)


def __getattr__(name):
    from ..implementation.triton.mem_cache import fused_sigmoid_mul as implementation

    return getattr(implementation, name)
