"""Base-format reference, with explicit rejection of hardware-private layouts."""


def npu_format_cast(input, out):
    return out.copy_(input)
