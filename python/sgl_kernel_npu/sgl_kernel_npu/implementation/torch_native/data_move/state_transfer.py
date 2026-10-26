"""Prepared physical-layout state copies; no boundary validation here."""


def all_layer_d2h(views, pairs):
    for device, host in views:
        for device_slot, host_slot in pairs:
            host[host_slot].copy_(device[:, device_slot], non_blocking=True)


def per_layer_h2d(source, destination, pairs):
    for device_slot, host_slot in pairs:
        destination[device_slot].copy_(source[host_slot], non_blocking=True)


def mamba_state(device, host, pairs, direction):
    for device_slot, host_slot in pairs:
        if direction == 1:
            device[:, device_slot].copy_(host[host_slot], non_blocking=True)
        else:
            host[host_slot].copy_(device[:, device_slot], non_blocking=True)
