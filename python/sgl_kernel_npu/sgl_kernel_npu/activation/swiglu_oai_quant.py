"""Compatibility imports for the relocated implementation."""

from ..operators.activation.swiglu_oai_quant import _align_up, swiglu_oai_quant


def __getattr__(name):
    from ..implementation.triton.activation import swiglu_oai_quant as implementation

    return getattr(implementation, name)
