"""Reference input validation and physical-storage view preparation.

CPU tensors are supported for reference diagnostics. Real host/NPU copies use
Tensor.copy_(non_blocking=True); ACL scheduling is not reproduced by the reference.
"""

from math import prod

import torch


def index_pairs(device_indices, host_indices, device_limit=None, host_limit=None):
    if device_indices.numel() != host_indices.numel():
        raise ValueError("device and host indices must have equal lengths")
    pairs = list(
        zip(
            device_indices.cpu().to(torch.int64).contiguous().reshape(-1).tolist(),
            host_indices.cpu().to(torch.int64).contiguous().reshape(-1).tolist(),
        )
    )
    if device_limit is not None and any(
        not 0 <= d < device_limit or not 0 <= h < host_limit for d, h in pairs
    ):
        raise ValueError("transfer index is outside the buffer slot range")
    return pairs


def validate_pairs(pairs, device_limit, host_limit):
    if any(not 0 <= d < device_limit or not 0 <= h < host_limit for d, h in pairs):
        raise ValueError("transfer index is outside the buffer slot range")


def dense_payload(tensor, start_dim):
    expected = 1
    dims = sorted(
        (tensor.stride(i), tensor.shape[i])
        for i in range(start_dim, tensor.ndim)
        if tensor.shape[i] > 1
    )
    for stride, size in dims:
        if stride != expected:
            raise ValueError("slot payload must be physically dense")
        expected *= size
    return prod(tensor.shape[start_dim:])


def all_layer_views(device, host):
    if device.ndim < 3 or host.ndim != device.ndim + 1:
        raise ValueError(
            "expected device [layers, slots, ...] and host [slots, layers, 1, ...]"
        )
    if not device.numel() or not host.numel() or device.dtype != host.dtype:
        raise ValueError("device/host states must be nonempty and have the same dtype")
    if (
        not host.is_contiguous()
        or host.shape[2] != 1
        or device.shape[0] != host.shape[1]
        or device.shape[2:] != host.shape[3:]
    ):
        raise ValueError(
            "host state layout must match the device payload and layer count"
        )
    payload = dense_payload(device, 2)
    if (
        device.stride(1) != payload
        or device.stride(0) < device.shape[1] * payload
        or host.stride(1) != payload
    ):
        raise ValueError("state slot/layer pitches do not match the native layout")
    # Native ACL copies physical bytes, including a densely permuted payload.
    return (
        device.as_strided(
            (device.shape[0], device.shape[1], payload),
            (device.stride(0), device.stride(1), 1),
        ),
        host.view(host.shape[0], host.shape[1], payload),
    )


def per_layer_views(src, dst, layer_id):
    if dst.ndim < 2 or src.ndim != dst.ndim + 2:
        raise ValueError(
            "expected host [slots, layers, 1, ...] and device [slots, ...]"
        )
    if not src.numel() or not dst.numel() or src.dtype != dst.dtype:
        raise ValueError("source/destination must be nonempty and have the same dtype")
    if not src.is_contiguous() or src.shape[2] != 1 or dst.shape[1:] != src.shape[3:]:
        raise ValueError("host layout must match the device layer payload")
    if not 0 <= layer_id < src.shape[1]:
        raise ValueError("layer_id is outside the host state layer range")
    payload = dense_payload(dst, 1)
    if dst.stride(0) != payload or src.stride(1) != payload:
        raise ValueError("state slot/layer pitches do not match the native layout")
    return (
        src.view(src.shape[0], src.shape[1], payload)[:, layer_id],
        dst.as_strided((dst.shape[0], payload), (dst.stride(0), 1)),
    )


def mamba_views(device, host):
    if device.ndim < 2 or host.ndim < 2 or not device.numel() or not host.numel():
        raise ValueError("state buffers must be nonempty with at least two dimensions")
    if (
        device.dtype != host.dtype
        or device.shape[0] != host.shape[1]
        or prod(device.shape[2:]) != prod(host.shape[2:])
    ):
        raise ValueError("state buffers must match in dtype, layers and payload size")
    if not device.is_contiguous() or not host.is_contiguous():
        raise ValueError("mamba transfer requires contiguous physical buffers")
    return device.view(device.shape[0], device.shape[1], -1), host.view(
        host.shape[0], host.shape[1], -1
    )


def kv_views(device, host, page_size):
    if device.ndim != 5 or host.ndim != 5 or not device.numel() or not host.numel():
        raise ValueError("KV buffers must be nonempty 5-D tensors")
    if (
        device.dtype != host.dtype
        or device.shape[0] != host.shape[1]
        or device.shape[2:] != host.shape[2:]
        or device.shape[2] != page_size
    ):
        raise ValueError(
            "device/host KV dtype, layer count and page payload must match"
        )
    if not device.is_contiguous() or not host.is_contiguous():
        raise ValueError("KV transfer requires contiguous physical buffers")
    return device, host
