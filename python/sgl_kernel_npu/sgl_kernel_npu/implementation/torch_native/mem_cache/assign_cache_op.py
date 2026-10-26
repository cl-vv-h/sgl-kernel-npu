"""Reference for the native boolean-returning cache assignment contract."""


def assign_cache_op(out, src, dst_start_idx, dst_end_idx, src_start_idx, src_end_idx):
    inputs = (out, src, dst_start_idx, dst_end_idx, src_start_idx, src_end_idx)
    if any(t.ndim < 1 for t in inputs):
        return False
    # Preserve the existing host-side batch checks, including its src size check.
    if not (
        out.shape[0]
        == src.shape[0]
        == dst_start_idx.shape[0]
        == dst_end_idx.shape[0]
        == src_start_idx.shape[0]
        == src_end_idx.shape[0]
    ):
        return False
    for row in range(out.shape[0]):
        start, end = int(dst_start_idx[row].item()), int(dst_end_idx[row].item())
        src_start, src_end = int(src_start_idx[row].item()), int(
            src_end_idx[row].item()
        )
        # The kernel copies the source interval length; dst_end_idx is not read for sizing.
        out[row, start : start + src_end - src_start].copy_(
            src.reshape(-1)[src_start:src_end]
        )
    return True
