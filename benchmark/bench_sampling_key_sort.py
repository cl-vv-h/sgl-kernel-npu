#!/usr/bin/env python3
"""Compare packed-key sort implementations without modifying installed packages.

Run on the same NPU/software stack as SGLang. By default this measures key sorting; --full-top-p adds complete top-p.
Neither measures MTP or TPOT. Both paths must preserve every FP32 key bit.
"""
import argparse
import json
import os
import statistics
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 16, 54])
    parser.add_argument("--vocab-size", type=int, default=248320)
    parser.add_argument("--draft-tokens", type=int, default=4)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--full-top-p",
        action="store_true",
        help="also time complete top-p renormalization",
    )
    args = parser.parse_args()
    if (
        min(args.batch_sizes) < 1
        or args.draft_tokens < 1
        or not 1 <= args.vocab_size <= 2**23
        or args.iterations < 1
        or args.warmup < 0
    ):
        parser.error("Invalid shape or iteration count")

    import torch
    import torch_npu

    if not hasattr(torch_npu, "npu_sort_v2"):
        raise RuntimeError("This torch_npu build does not expose npu_sort_v2")
    torch.npu.set_device(args.device)
    torch.manual_seed(20261010)
    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "torch_npu": torch_npu.__version__,
                "device": torch.npu.get_device_name(),
                "args": vars(args),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    for batch in args.batch_sizes:
        rows = batch * args.draft_tokens
        # The real packed-key input is a permutation of 2*token_id + keep_bit.
        # Construct the same contract, including large odd keys that reveal
        # lossy FP16/BF16 conversion. Generation is outside the timing window.
        with torch.inference_mode():
            indices = torch.rand((rows, args.vocab_size), device=args.device).argsort(
                -1
            )
            keys = indices.to(torch.float32).mul_(2)
            del indices
            bits = torch.randint(
                0, 2, keys.shape, device=args.device, dtype=torch.int32
            )
            keys.add_(bits)
            del bits
            paths = {
                "torch_sort_values": lambda: keys.sort(dim=-1).values,
                "npu_sort_v2": lambda: torch_npu.npu_sort_v2(
                    keys, dim=-1, descending=False
                ),
            }
            expected = paths["torch_sort_values"]()
            actual = paths["npu_sort_v2"]()
            torch.npu.synchronize()
            if actual.dtype != torch.float32:
                raise AssertionError(f"Output dtype changed: {actual.dtype}")
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            del actual, expected
            print(
                json.dumps(
                    {"batch": batch, "rows": rows, "check": "EXACT_KEYS_PASSED"}
                ),
                flush=True,
            )
            for _ in range(args.warmup):
                for fn in paths.values():
                    output = fn()
                    torch.npu.synchronize()
                    del output
            durations = {name: [] for name in paths}
            names = list(paths)
            for iteration in range(args.iterations):
                for name in names[:: 1 if iteration % 2 == 0 else -1]:
                    torch.npu.synchronize()
                    start = time.perf_counter()
                    output = paths[name]()
                    torch.npu.synchronize()
                    durations[name].append((time.perf_counter() - start) * 1000)
                    del output
            medians = {
                name: statistics.median(vals) for name, vals in durations.items()
            }
            print(
                json.dumps(
                    {
                        "batch": batch,
                        "rows": rows,
                        "vocab": args.vocab_size,
                        "median_ms": medians,
                        "saved_ms": medians["torch_sort_values"]
                        - medians["npu_sort_v2"],
                        "scope": "key sort only; not full top-p or TPOT",
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            del keys
            if args.full_top_p:
                from sgl_kernel_npu.sample.probability import top_p_renorm_prob

                os.environ["SGL_KERNEL_NPU_SAMPLING_TRITON"] = "1"
                os.environ["SGL_KERNEL_NPU_SAMPLING_TRITON_DIV"] = "rn"
                probs = torch.softmax(
                    torch.randn(rows, args.vocab_size, device=args.device), dim=-1
                )
                top_ps = torch.full((rows,), 0.95, device=args.device)
                timings = {"v1_rn": [], "values_sort": []}
                expected = None
                for iteration in range(args.warmup + args.iterations):
                    for name in list(timings)[:: 1 if iteration % 2 == 0 else -1]:
                        os.environ["SGL_KERNEL_NPU_SAMPLING_VALUES_SORT"] = str(
                            int(name == "values_sort")
                        )
                        torch.npu.synchronize()
                        start = time.perf_counter()
                        output = top_p_renorm_prob(probs, top_ps)
                        torch.npu.synchronize()
                        if iteration >= args.warmup:
                            timings[name].append((time.perf_counter() - start) * 1000)
                        if expected is None:
                            expected = output
                        else:
                            torch.testing.assert_close(output, expected, rtol=0, atol=0)
                medians = {name: statistics.median(t) for name, t in timings.items()}
                print(
                    json.dumps(
                        dict(
                            batch=batch,
                            rows=rows,
                            check="EXACT_TOP_P_PASSED",
                            median_ms=medians,
                            saved_ms=medians["v1_rn"] - medians["values_sort"],
                            scope="complete top-p, not TPOT",
                        )
                    ),
                    flush=True,
                )
                del probs, output, expected
    print("ALL_KEY_CHECKS_PASSED", flush=True)


if __name__ == "__main__":
    main()
