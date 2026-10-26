"""Tensor-backed indexed copies; opaque device pointers require the native kernel."""


def unidex_copy(src, dst, src_index, dst_index):
    dst[dst_index] = src[src_index].to(dst.device)
