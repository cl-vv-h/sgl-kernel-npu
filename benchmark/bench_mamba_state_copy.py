"""Check state-copy equality and compare original / tiled device execution."""

import argparse
import json
import os
import statistics
import time


def make_case(batch, layers, heads, width, keys, draft, dtype, strided=False):
    import torch

    torch.manual_seed(17)
    src = torch.randn(
        (layers, batch, draft, heads, width, keys), device="npu", dtype=dtype
    )
    if strided:
        src = torch.randn(
            (layers, batch, draft, heads, keys, width), device="npu", dtype=dtype
        ).transpose(-1, -2)
        dst = torch.full(
            (layers, batch * 2, heads, keys, width), -3.0, device="npu", dtype=dtype
        ).transpose(-1, -2)
    else:
        dst = torch.full(
            (layers, batch * 2, heads, width, keys), -3.0, device="npu", dtype=dtype
        )
    di = torch.arange(batch - 1, -1, -1, dtype=torch.int64, device="npu") * 2
    si = torch.arange(batch - 1, -1, -1, dtype=torch.int64, device="npu")
    steps = torch.arange(batch, dtype=torch.int64, device="npu") % (draft + 1) - 1
    return src, dst, di, si, steps


def check_case(batch, layers, heads, width, keys, draft, dtype, block, strided=False):
    import torch
    from sgl_kernel_npu.mamba.mamba_state_update_triton import move_intermediate_cache

    src, dst, di, si, steps = make_case(
        batch, layers, heads, width, keys, draft, dtype, strided
    )
    expected = dst.clone()
    # CPU loop is outside measured regions and independently defines semantics.
    for d, s, step in zip(di.cpu().tolist(), si.cpu().tolist(), steps.cpu().tolist()):
        if step >= 0:
            expected[:, d].copy_(src[:, s, step])
    os.environ["SGL_KERNEL_NPU_STATE_COPY_TILED"] = "1"
    os.environ["SGL_KERNEL_NPU_STATE_COPY_BLOCK"] = str(block)
    move_intermediate_cache(dst, src, di, si, steps)
    torch.testing.assert_close(dst, expected, rtol=0, atol=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 16, 54])
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dim-v", type=int, default=128)
    parser.add_argument("--dim-k", type=int, default=128)
    parser.add_argument("--draft-tokens", type=int, default=4)
    parser.add_argument(
        "--blocks", nargs="+", type=int, default=[1024, 2048, 4096, 8192]
    )
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    if (
        min(
            *args.batch_sizes,
            args.layers,
            args.heads,
            args.dim_v,
            args.dim_k,
            args.draft_tokens,
            args.iterations
        )
        < 1
        or args.warmup < 0
        or any(b not in (1024, 2048, 4096, 8192) for b in args.blocks)
    ):
        parser.error("invalid shape, block size or iteration count")
    import torch
    import torch_npu
    from sgl_kernel_npu.mamba.mamba_state_update_triton import move_intermediate_cache

    print(
        json.dumps(
            dict(
                torch=torch.__version__,
                torch_npu=torch_npu.__version__,
                args=vars(args),
            )
        ),
        flush=True,
    )
    dtype = getattr(torch, args.dtype)
    for block in args.blocks:
        check_case(5, 3, 3, 17, 31, 4, dtype, block, strided=True)
    for batch in args.batch_sizes:
        src, dst, di, si, steps = make_case(
            batch,
            args.layers,
            args.heads,
            args.dim_v,
            args.dim_k,
            args.draft_tokens,
            dtype,
        )
        # Time active copies for B=1 too. Dedicated checks cover negative steps.
        steps.fill_(args.draft_tokens - 1)
        baseline = dst.clone()
        os.environ["SGL_KERNEL_NPU_STATE_COPY_TILED"] = "0"
        move_intermediate_cache(baseline, src, di, si, steps)
        for block in args.blocks:
            os.environ["SGL_KERNEL_NPU_STATE_COPY_BLOCK"] = str(block)
            os.environ["SGL_KERNEL_NPU_STATE_COPY_TILED"] = "1"
            move_intermediate_cache(dst, src, di, si, steps)
            torch.testing.assert_close(dst, baseline, rtol=0, atol=0)
            times = {"original": [], "tiled": []}
            for i in range(args.warmup + args.iterations):
                for name in list(times)[:: 1 if i % 2 == 0 else -1]:
                    os.environ["SGL_KERNEL_NPU_STATE_COPY_TILED"] = str(
                        int(name == "tiled")
                    )
                    torch.npu.synchronize()
                    start = time.perf_counter()
                    move_intermediate_cache(dst, src, di, si, steps)
                    torch.npu.synchronize()
                    if i >= args.warmup:
                        times[name].append((time.perf_counter() - start) * 1000)
            ms = {name: statistics.median(t) for name, t in times.items()}
            print(
                json.dumps(
                    dict(
                        batch=batch,
                        block=block,
                        check="EXACT_STATE_PASSED",
                        median_ms=ms,
                        saved_ms=ms["original"] - ms["tiled"],
                        scope="state copy only, not TPOT",
                    )
                ),
                flush=True,
            )
        del src, dst, baseline


if __name__ == "__main__":
    main()
