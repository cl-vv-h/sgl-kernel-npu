"""Reference page copies over prepared device/host views."""


def transfer_kv_dim_exchange(views, pages, direction):
    for device, host in views:
        for device_page, host_page in pages:
            if direction == 1:
                device[:, device_page].copy_(host[host_page], non_blocking=True)
            else:
                host[host_page].copy_(device[:, device_page], non_blocking=True)
