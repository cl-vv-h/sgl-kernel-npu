"""Reference for the native boolean-returning cache assignment contract."""


def assign_cache_op(out, src, dst_start_idx, src_start_idx, src_end_idx):
    for row in range(out.shape[0]):
        start = dst_start_idx[row]
        src_start, src_end = src_start_idx[row], src_end_idx[row]
        # The kernel copies the source interval length; dst_end_idx is not read for sizing.
        out[row, start : start + src_end - src_start].copy_(src[src_start:src_end])
    return True
