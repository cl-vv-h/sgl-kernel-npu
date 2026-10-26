import torch


def npu_log_softmax_batch_invariant(input, dim):
    # Mathematical reference; no promise of bitwise batch-invariant reduction order.
    return torch.log_softmax(input.float(), dim=dim).to(input.dtype)
