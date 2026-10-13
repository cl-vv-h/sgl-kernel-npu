"""Paired scan/filter benchmark against fcf14ce V1+RN, without installing source.

The service candidate retains native probability/key sorting. Probability merge
is diagnostic-only and opt-in after the reported packed-key merge regression.
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
from sgl_kernel_npu.utils.triton_utils import get_device_properties


def native_prepare(values, indices, thresholds):
    rows, vocab = values.shape
    blocks = triton.cdiv(vocab, 2048)
    grid = (min(rows * blocks, get_device_properties()[1]),)
    cumulative = values.cumsum(-1)
    keys = torch.empty_like(values)
    op._filter_encode_kernel[grid](
        values,
        indices,
        cumulative,
        thresholds,
        keys,
        rows,
        vocab,
        blocks,
        thresholds.stride(0),
        True,
        2048,
    )
    return keys


def baseline(probs, thresholds):
    values, indices = probs.sort(-1, descending=True)
    return op.filter_and_renorm(probs, values, indices, thresholds, values.cumsum(-1))


def capture(fn):
    for _ in range(3):
        fn()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, auto_dispatch_capture=True):
        output = fn()
    return graph.replay, graph, output


def paired(case, old, new, args, batch, tile):
    owners = []
    if args.graph:
        old, graph, output = capture(old)
        owners.extend([graph, output])
        new, graph, output = capture(new)
        owners.extend([graph, output])
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
    a, b = map(statistics.median, times)
    print(
        json.dumps(
            dict(
                case=case,
                batch=batch,
                rows=batch * args.draft_tokens,
                vocab=args.vocab_size,
                tile=tile,
                graph=args.graph,
                baseline_ms=a,
                candidate_ms=b,
                paired_saved_ms=statistics.median([x - y for x, y in zip(*times)]),
                speedup=a / b,
            )
        ),
        flush=True,
    )


def difference(actual, expected):
    a, b = actual.detach().cpu(), expected.detach().cpu()
    mismatches = int((a != b).sum())
    keep = int(((a != 0) != (b != 0)).sum())
    gen = torch.Generator().manual_seed(93)
    coins = torch.rand(a.shape[0], 8, generator=gen)
    # Categorical diagnostic only; this is not an MTP acceptance-length test.
    sampled_a = torch.searchsorted(a.cumsum(-1).contiguous(), coins, right=True)
    sampled_b = torch.searchsorted(b.cumsum(-1).contiguous(), coins, right=True)
    return dict(
        probability_mismatches=mismatches,
        keep_mismatches=keep,
        max_absolute_error=float((a - b).abs().max()),
        categorical_mismatches=int((sampled_a != sampled_b).sum()),
        v1_exact=mismatches == 0,
    )


def precision_check(probs, thresholds, tile, args, label):
    # Share the native permutation to isolate scan/filter rounding differences.
    values, indices = probs.sort(-1, descending=True)
    native_values = values.clone()
    fused_values = values.clone()
    native_keys = native_prepare(native_values, indices, thresholds)
    fused_keys = op._scan_filter_and_encode(fused_values, indices, thresholds, tile)
    expected = op.filter_and_renorm(
        probs, values.clone(), indices, thresholds, values.cumsum(-1)
    )
    actual = op._top_p_renorm_prob_scan_fused(probs, thresholds, tile)
    report = difference(actual, expected)
    report.update(
        check=label,
        tile=tile,
        key_mismatches=int((native_keys != fused_keys).sum().item()),
    )
    print(json.dumps(report), flush=True)
    if args.require_v1_exact and (not report["v1_exact"] or report["key_mismatches"]):
        raise AssertionError("V1 exactness gate failed; do not install this candidate.")
    return report["v1_exact"] and report["key_mismatches"] == 0


def main():
    global op
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
    parser.add_argument(
        "--tiles", type=int, nargs="+", choices=[512, 1024, 2048], default=[2048]
    )
    parser.add_argument(
        "--probability-tile", type=int, choices=[256, 512, 1024], default=512
    )
    parser.add_argument("--diagnose-probability-merge", action="store_true")
    parser.add_argument(
        "--modes", nargs="+", choices=["scan", "sort-int32"], default=["scan"]
    )
    parser.add_argument("--require-v1-exact", action="store_true")
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--graph", action="store_true")
    args = parser.parse_args()
    if (
        min(args.batch_sizes) < 1
        or min(args.vocab_size, args.draft_tokens, args.iterations) < 1
    ):
        parser.error("sizes and iterations must be positive")
    if args.temperature <= 0:
        parser.error("temperature must be positive")
    if not 0 <= args.top_p <= 1:
        parser.error("top-p must be in [0,1]")
    if args.diagnose_probability_merge and "sort-int32" not in args.modes:
        args.modes.append("sort-int32")
    torch.npu.set_device(args.device)
    spec = importlib.util.spec_from_file_location("scan_candidate", args.source)
    op = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(op)
    print(
        json.dumps(
            dict(
                args={**vars(args), "source": str(args.source)},
                source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
                torch=torch.__version__,
                torch_npu=torch_npu.__version__,
                triton=triton.__version__,
                device=torch.npu.get_device_name(),
                baseline="fcf14ce V1+RN",
                division="rn",
                service_probability_sort="native",
                service_key_sort="native",
            )
        ),
        flush=True,
    )
    exact = True
    torch.manual_seed(91)
    with torch.inference_mode():
        # Endpoints, exact ties/zeros and thresholds at an ULP boundary.
        for kind in (
            "random",
            "bf16_ties",
            "uniform",
            "zeros",
            "one_hot",
            "ulp_boundary",
        ):
            probs = torch.randn(4, 2051, device=args.device)
            if kind == "bf16_ties":
                probs = probs.bfloat16().float()
            probs = (probs / args.temperature).softmax(-1)
            ps = torch.tensor([0, 0.25, 0.95, 1], device=args.device)
            if kind == "uniform":
                probs.fill_(1 / 2051)
            elif kind == "zeros":
                probs.zero_()
            elif kind == "one_hot":
                probs.zero_()
                probs[:, -1] = 1
            elif kind == "ulp_boundary":
                values = probs.sort(-1, descending=True).values
                boundary = (values.cumsum(-1) - values)[:, 1024]
                ps = torch.nextafter(boundary, torch.full_like(boundary, -float("inf")))
            if "scan" in args.modes:
                for tile in args.tiles:
                    exact &= precision_check(probs, ps, tile, args, kind)
        for batch in args.batch_sizes:
            rows = batch * args.draft_tokens
            probs = (
                torch.randn(rows, args.vocab_size, device=args.device)
                .bfloat16()
                .float()
            )
            probs = (probs / args.temperature).softmax(-1)
            ps = torch.full((rows,), args.top_p, device=args.device)
            values, indices = probs.sort(-1, descending=True)
            if "scan" in args.modes:
                for tile in args.tiles:
                    exact &= precision_check(
                        probs, ps, tile, args, f"full/batch={batch}"
                    )
                    paired(
                        "scan_filter_including_clone",
                        lambda: native_prepare(values.clone(), indices, ps),
                        lambda: op._scan_filter_and_encode(
                            values.clone(), indices, ps, tile
                        ),
                        args,
                        batch,
                        tile,
                    )
                    paired(
                        "full_top_p",
                        lambda: baseline(probs, ps),
                        lambda: op._top_p_renorm_prob_scan_fused(probs, ps, tile),
                        args,
                        batch,
                        tile,
                    )
            if "sort-int32" in args.modes:
                pv, pi, _ = op._sort_probabilities_int32(probs, args.probability_tile)
                torch.testing.assert_close(pv, values, rtol=0, atol=0)
                ordered = torch.gather(probs, -1, pi.long())
                torch.testing.assert_close(pv, ordered, rtol=0, atol=0)
                torch.testing.assert_close(
                    pi.long().sort(-1).values,
                    torch.arange(args.vocab_size, device=args.device).expand(rows, -1),
                    rtol=0,
                    atol=0,
                )
                print(
                    json.dumps(
                        dict(
                            check="diagnostic_probability_merge",
                            native_identity_mismatches=int(
                                (pi != indices).sum().item()
                            ),
                            active_in_service=False,
                        )
                    ),
                    flush=True,
                )
                expected = baseline(probs, ps)
                actual = op._top_p_renorm_prob_sort_int32(
                    probs, ps, args.probability_tile
                )
                report = difference(actual, expected)
                report.update(
                    check="probability_sort_full_top_p",
                    batch=batch,
                    tile=args.probability_tile,
                    active_in_service=False,
                )
                print(json.dumps(report), flush=True)
                exact &= report["v1_exact"]
                if args.require_v1_exact and not report["v1_exact"]:
                    raise AssertionError(
                        "INT32 sort changes V1 outputs; do not deploy it."
                    )
                del actual, expected, pv, pi, ordered
                paired(
                    "diagnostic_probability_sort",
                    lambda: probs.sort(-1, descending=True),
                    lambda: op._sort_probabilities_int32(probs, args.probability_tile),
                    args,
                    batch,
                    args.probability_tile,
                )
                paired(
                    "diagnostic_sort_full_top_p",
                    lambda: baseline(probs, ps),
                    lambda: op._top_p_renorm_prob_sort_int32(
                        probs, ps, args.probability_tile
                    ),
                    args,
                    batch,
                    args.probability_tile,
                )

    print(
        json.dumps(
            dict(
                measured_cases_v1_exact=bool(exact),
                note="Positive paired_saved_ms means faster. Exactness over finite cases is not a universal proof.",
            )
        )
    )


if __name__ == "__main__":
    main()
