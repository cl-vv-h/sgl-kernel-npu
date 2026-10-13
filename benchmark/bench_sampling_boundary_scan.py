"""Paired V1+RN / boundary / scan / combined benchmark, without installation.

Loads this checkout's probability.py directly. Native-scan boundary restoration
must be bitwise equivalent to V1; tiled-scan rounding differences are reported
separately. These synthetic tests do not establish model accuracy or TPOT.
"""

import argparse
import hashlib
import importlib.util
import json
import statistics
import time
from pathlib import Path

import torch
import torch_npu
import triton

MODES = {"boundary": (True, False), "scan": (False, True), "both": (True, True)}


def load_source(path):
    spec = importlib.util.spec_from_file_location(
        "sampling_boundary_scan_candidate", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_sorted(op, probs, values, indices, thresholds, boundary=False, scan=False):
    cumulative = None if scan else values.cumsum(-1)
    return op._renorm_sorted_experimental(
        probs,
        values,
        indices,
        thresholds,
        cumulative,
        boundary_restore=boundary,
        scan_fusion=scan,
    )


def run_full(op, probs, thresholds, boundary=False, scan=False):
    values, indices = probs.sort(-1, descending=True)
    return run_sorted(op, probs, values, indices, thresholds, boundary, scan)


def difference(actual, expected):
    actual, expected = actual.detach().cpu(), expected.detach().cpu()
    mismatches = int((actual != expected).sum())
    return {
        "probability_mismatches": mismatches,
        "keep_mismatches": int(((actual != 0) != (expected != 0)).sum()),
        "max_absolute_error": float((actual - expected).abs().max()),
        "exact": mismatches == 0,
    }


def correction_stats(op, values, indices, thresholds, scan):
    # Diagnostic only: all these launches/CPU transfers are outside timing.
    rows, vocab = values.shape
    blocks = triton.cdiv(vocab, 2048)
    cores = op.get_device_properties()[1]
    grid = (min(rows * blocks, cores),)
    row_grid = (min(rows, cores),)
    work = values.clone()
    if scan:
        totals = torch.empty((rows, blocks), device=values.device, dtype=torch.float32)
        op._tile_totals_kernel[grid](work, totals, rows, vocab, blocks, 2048)
        prefixes = totals.cumsum(-1)
    else:
        prefixes = values.cumsum(-1)
    bounds = torch.empty((rows, blocks, 4), device=values.device, dtype=torch.float32)
    op._experimental_filter_kernel[grid](
        work,
        indices,
        prefixes,
        thresholds,
        bounds,
        rows,
        vocab,
        blocks,
        thresholds.stride(0),
        True,
        scan,
        True,
        2048,
    )
    metadata = torch.empty((rows, 4), device=values.device, dtype=torch.float32)
    op._boundary_metadata_kernel[row_grid](
        bounds, metadata, rows, blocks, triton.next_power_of_2(blocks)
    )
    metadata = metadata.cpu()
    begin, end = metadata[:, 1], metadata[:, 2]
    tiles = (end - begin).clamp_min(0)
    slots = ((end * 2048).clamp_max(vocab) - begin * 2048).clamp_min(0)
    return {
        "correction_tiles_per_row_mean": float(tiles.mean()),
        "correction_tiles_per_row_max": int(tiles.max()),
        "correction_slots_per_row_mean": float(slots.mean()),
        "correction_slots_per_row_max": int(slots.max()),
        "correction_fraction": float(slots.sum()) / (rows * vocab),
        "total_tiles_per_row": blocks,
    }


def check(op, probs, thresholds, modes, require_exact, label):
    values, indices = probs.sort(-1, descending=True)
    original = probs.clone()
    expected = run_sorted(op, probs, values.clone(), indices, thresholds)
    exact = True
    for mode in modes:
        boundary, scan = MODES[mode]
        work = values.clone()
        actual = run_sorted(op, probs, work, indices, thresholds, boundary, scan)
        torch.testing.assert_close(probs, original, rtol=0, atol=0)
        # Compare to scatter using the candidate's actual filtered values. This
        # separates restoration errors from changed prefix rounding in scan mode.
        filtered = work / work.sum(-1, keepdim=True).clamp_min(1e-20)
        restored_reference = torch.zeros_like(probs).scatter_(-1, indices, filtered)
        torch.testing.assert_close(actual, restored_reference, rtol=0, atol=0)
        report = {
            "kind": "correctness",
            "check": label,
            "mode": mode,
            "rows": probs.shape[0],
            "vocab": probs.shape[1],
            "restoration_exact": True,
            **difference(actual, expected),
        }
        if boundary:
            report.update(correction_stats(op, values, indices, thresholds, scan))
        print(json.dumps(report), flush=True)
        exact = exact and report["exact"]
        if not scan and not report["exact"]:
            raise AssertionError("Native-scan boundary restoration differs from V1.")
        if require_exact and not report["exact"]:
            raise AssertionError("V1 exactness gate failed; do not deploy this mode.")
    return exact


def capture(fn):
    for _ in range(3):
        fn()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, auto_dispatch_capture=True):
        output = fn()
    torch.npu.synchronize()
    return graph.replay, (graph, output)


def paired(old, new, args, mode, batch, case):
    owners = []
    if args.graph:
        old, owner = capture(old)
        owners.append(owner)
        new, owner = capture(new)
        owners.append(owner)
    for _ in range(args.warmup):
        old()
        new()
    torch.npu.synchronize()
    times = [[], []]
    for iteration in range(args.iterations):
        for which in (0, 1) if iteration % 2 == 0 else (1, 0):
            torch.npu.synchronize()
            start = time.perf_counter()
            (old if which == 0 else new)()
            torch.npu.synchronize()
            times[which].append((time.perf_counter() - start) * 1000)
    baseline, candidate = map(statistics.median, times)
    print(
        json.dumps(
            {
                "kind": "timing",
                "case": case,
                "mode": mode,
                "batch": batch,
                "rows": batch * args.draft_tokens,
                "vocab": args.vocab_size,
                "graph": args.graph,
                "iterations": args.iterations,
                "baseline_ms": baseline,
                "candidate_ms": candidate,
                "paired_saved_ms": statistics.median([a - b for a, b in zip(*times)]),
                "speedup": baseline / candidate,
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "python/sgl_kernel_npu/sgl_kernel_npu/sample/probability.py",
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[10, 20, 33, 54])
    parser.add_argument("--vocab-size", type=int, default=248320)
    parser.add_argument("--draft-tokens", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--modes", nargs="+", choices=list(MODES), default=list(MODES))
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--breakdown", action="store_true")
    parser.add_argument("--require-v1-exact", action="store_true")
    args = parser.parse_args()
    if (
        min(args.batch_sizes) < 1
        or args.draft_tokens < 1
        or args.iterations < 1
        or args.warmup < 0
    ):
        parser.error(
            "batch sizes/draft tokens/iterations must be positive; warmup must be nonnegative"
        )
    if (
        not 1 <= args.vocab_size <= 2**23
        or args.temperature <= 0
        or not 0 <= args.top_p <= 1
    ):
        parser.error("invalid vocabulary, temperature or top-p")
    torch.npu.set_device(args.device)
    op = load_source(args.source.resolve())
    print(
        json.dumps(
            {
                "kind": "metadata",
                "args": {**vars(args), "source": str(args.source.resolve())},
                "sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
                "torch": torch.__version__,
                "torch_npu": torch_npu.__version__,
                "triton": triton.__version__,
                "device": torch.npu.get_device_name(),
                "baseline": "fcf14ce / V1+RN",
                "division": "rn",
                "import_switches": {
                    "boundary": op._BOUNDARY_RESTORE,
                    "scan": op._SCAN_FUSION,
                },
                "benchmark_dispatch": "explicit helpers; does not modify installed package",
            }
        ),
        flush=True,
    )
    edge = torch.tensor(
        [
            [0.5, 0.25, 0.125, 0.125],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0.25, 0.25, 0.25, 0.25],
            [1e-25, 2e-25, 3e-25, 4e-25],
        ],
        device=args.device,
        dtype=torch.float32,
    )
    edge_ps = torch.tensor([0, 0.95, 1, 0.5, 0.95], device=args.device)
    all_exact = check(op, edge, edge_ps, args.modes, args.require_v1_exact, "edges")
    # Thresholds around a rounded prefix expose differences rather than hiding
    # them behind an error tolerance. Without --require-v1-exact, timing proceeds.
    boundary_probs = torch.softmax(torch.randn(3, 2051, device=args.device), -1)
    sorted_probs = boundary_probs.sort(-1, descending=True).values
    cutoffs = sorted_probs.cumsum(-1)[:, 1024] - sorted_probs[:, 1024]
    all_exact = (
        check(
            op,
            boundary_probs,
            cutoffs,
            args.modes,
            args.require_v1_exact,
            "rounding_boundary",
        )
        and all_exact
    )
    for batch in args.batch_sizes:
        rows = batch * args.draft_tokens
        torch.manual_seed(1)
        logits = (
            torch.randn(rows, args.vocab_size, device=args.device)
            .to(torch.bfloat16)
            .float()
        )
        probs = torch.softmax(logits / args.temperature, -1)
        thresholds = torch.full((rows,), args.top_p, device=args.device)
        all_exact = (
            check(
                op,
                probs,
                thresholds,
                args.modes,
                args.require_v1_exact,
                f"batch_{batch}",
            )
            and all_exact
        )
        values, indices = probs.sort(-1, descending=True)
        for mode in args.modes:
            boundary, scan = MODES[mode]
            paired(
                lambda: run_full(op, probs, thresholds),
                lambda: run_full(op, probs, thresholds, boundary, scan),
                args,
                mode,
                batch,
                "full_top_p",
            )
            if args.breakdown:
                # Both sides include the same clone cost; neither includes the
                # initial probability sort. Native cumsum is timed on its side.
                paired(
                    lambda: run_sorted(op, probs, values.clone(), indices, thresholds),
                    lambda: run_sorted(
                        op, probs, values.clone(), indices, thresholds, boundary, scan
                    ),
                    args,
                    mode,
                    batch,
                    "renorm_including_clone",
                )
    print(
        json.dumps(
            {
                "kind": "summary",
                "measured_cases_v1_exact": all_exact,
                "scope": "synthetic finite cases; no model-accuracy/accept-length claim",
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
