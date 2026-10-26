def npu_scatter_nd_update_(input, indices, updates):
    depth = indices.shape[-1]
    if depth == 0 or depth > input.ndim:
        raise ValueError("indices depth must be in [1, input.ndim]")
    coordinates = indices.long().reshape(-1, depth)
    input[tuple(coordinates[:, d] for d in range(depth))] = updates.reshape(
        coordinates.shape[0], *input.shape[depth:]
    )
    return input
