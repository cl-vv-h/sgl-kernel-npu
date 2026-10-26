def npu_reduce_mean_batch_invariant(input, dim, keepdim=False):
    # Mathematical reference; no promise of bitwise batch-invariant reduction order.
    return input.float().mean(dim=dim, keepdim=keepdim).to(input.dtype)
