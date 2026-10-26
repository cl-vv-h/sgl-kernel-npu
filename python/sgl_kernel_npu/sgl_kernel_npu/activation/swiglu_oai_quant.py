"""Compatibility imports for the relocated implementation."""

from ..operators.mem_cache.swiglu_oai_quant import swiglu_oai_quant


def __getattr__(name):
    from ..implementation.triton.mem_cache import swiglu_oai_quant as implementation

    return getattr(implementation, name)
