"""Compatibility imports for SiTU kernels and public adapters."""

from ..operators.activation.situ_and_mul import situ_and_mul


def __getattr__(name):
    from ..implementation.triton.activation import situ as implementation

    return getattr(implementation, name)


__all__ = ["situ_and_mul", "situ_and_mul_quant", "situ"]
