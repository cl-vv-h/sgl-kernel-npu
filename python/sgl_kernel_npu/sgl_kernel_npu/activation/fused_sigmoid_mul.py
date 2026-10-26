"""Compatibility imports for the relocated implementation."""

from ..implementation.triton.activation.fused_sigmoid_mul import fused_sigmoid_mul
from ..operators.activation.fused_sigmoid_mul_broadcast import (
    fused_sigmoid_mul_broadcast,
)


def __getattr__(name):
    from ..implementation.triton.activation import fused_sigmoid_mul as implementation

    return getattr(implementation, name)
