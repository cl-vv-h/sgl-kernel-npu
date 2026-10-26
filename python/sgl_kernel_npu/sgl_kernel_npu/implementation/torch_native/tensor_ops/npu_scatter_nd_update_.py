def npu_scatter_nd_update_(input, indices, updates):
    input[indices] = updates
    return input
