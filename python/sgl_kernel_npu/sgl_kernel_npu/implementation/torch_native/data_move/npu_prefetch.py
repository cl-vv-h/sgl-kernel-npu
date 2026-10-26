"""Prefetch has no equivalent cache-management primitive in standard PyTorch."""


def npu_prefetch(input, dependency, max_size, offset=0):
    raise NotImplementedError(
        "torch_native cannot reproduce NPU cache prefetch or its dependency scheduling; "
        "use impl='torch_npu'. A no-op would not implement this operator."
    )
