"""Paired NPU benchmark against the fixed V1+RN packed-key native-sort path.

No package installation or environment dispatch is performed by this script.
Run correctness tests before timing. First probability sort/cumsum and RN are
identical; only the encoded-key restore pipeline differs.
"""

import argparse
import importlib.util
import json
import statistics
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401
import triton
from sgl_kernel_npu.utils.triton_utils import get_device_properties


def baseline_restore(probs, values, indices, thresholds, cumulative):
    # V1+RN algorithm at fcf14ce. SORT_KEYS=False disables only the new local
    # sort in the shared filtering kernel; V1 used this same filtering body.
    rows, vocab = probs.shape
    blocks = triton.cdiv(vocab, 2048)
    grid = (min(rows * blocks, get_device_properties()[1]),)
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
        False,
    )
    denominator = values.sum(-1, keepdim=True).clamp_min_(1e-20)
    ordered = keys.sort(-1).values
    output = torch.empty_like(probs)
    op._restore_kernel[grid](
        probs, ordered, denominator, output, rows, vocab, blocks, 2048
    )
    return output


def full_top_p(probs, thresholds, tile=None):
    values, indices = probs.sort(-1, descending=True)
    cumulative = values.cumsum(-1)
    if tile is None:
        return baseline_restore(probs, values, indices, thresholds, cumulative)
    return op.filter_and_renorm(probs, values, indices, thresholds, cumulative, tile)


def capture(fn):
    for _ in range(3):
        fn()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, auto_dispatch_capture=True):
        output = fn()
    return graph.replay, graph, output


def paired(case, baseline, candidate, args, batch, tile):
    owners = []
    if args.graph:
        baseline, graph, output = capture(baseline)
        owners.extend([graph, output])
        candidate, graph, output = capture(candidate)
        owners.extend([graph, output])
    for _ in range(args.warmup):
        baseline()
        candidate()
    torch.npu.synchronize()
    times = [[], []]
    for iteration in range(args.iterations):
        for which in (0, 1) if iteration % 2 == 0 else (1, 0):
            torch.npu.synchronize()
            start = time.perf_counter()
            (baseline if which == 0 else candidate)()
            torch.npu.synchronize()
            times[which].append((time.perf_counter() - start) * 1000)
    old, new = map(statistics.median, times)
    print(
        json.dumps(
            dict(
                case=case,
                batch=batch,
                rows=batch * args.draft_tokens,
                vocab=args.vocab_size,
                tile=tile,
                graph=args.graph,
                baseline_ms=old,
                candidate_ms=new,
                paired_saved_ms=statistics.median([a - b for a, b in zip(*times)]),
                speedup=old / new,
            )
        ),
        flush=True,
    )


def main():
    global op
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[24, 54])
    parser.add_argument("--vocab-size", type=int, default=248320)
    parser.add_argument("--draft-tokens", type=int, default=4)
    parser.add_argument(
        "--tiles", type=int, nargs="+", choices=[512, 1024, 2048], default=[2048]
    )
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "python/sgl_kernel_npu/sgl_kernel_npu/sample/probability.py",
    )
    args = parser.parse_args()
    if (
        min(args.batch_sizes) < 1
        or min(args.vocab_size, args.draft_tokens, args.iterations) < 1
    ):
        parser.error("sizes and iterations must be positive")
    if not 0 <= args.top_p <= 1:
        parser.error("top-p must be in [0, 1]")
    torch.npu.set_device(args.device)
    spec = importlib.util.spec_from_file_location("packed_key_candidate", args.source)
    op = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(op)
    print(
        json.dumps(
            dict(
                args={**vars(args), "source": str(args.source)},
                torch=torch.__version__,
                torch_npu=torch_npu.__version__,
                triton=triton.__version__,
                source=op.__file__,
                baseline="fcf14ce V1+RN",
                device=torch.npu.get_device_name(),
            )
        ),
        flush=True,
    )
    torch.manual_seed(73)
    with torch.inference_mode():
        for batch in args.batch_sizes:
            rows = batch * args.draft_tokens
            # BF16-origin logits exercise ties; probabilities always remain FP32.
            logits = (
                torch.randn(rows, args.vocab_size, device=args.device)
                .bfloat16()
                .float()
            )
            probs = logits.softmax(-1)
            thresholds = torch.full((rows,), args.top_p, device=args.device)
            values, indices = probs.sort(-1, descending=True)
            cumulative = values.cumsum(-1)
            filtered = values.masked_fill(cumulative - values > thresholds[:, None], 0)
            keys = indices.float() * 2 + (filtered != 0).float()
            expected = baseline_restore(
                probs, values.clone(), indices, thresholds, cumulative
            )
            del filtered, logits
            for tile in args.tiles:
                ordered = op._sort_packed_keys(keys, tile)
                torch.testing.assert_close(
                    ordered, keys.sort(-1).values, rtol=0, atol=0
                )
                actual = op.filter_and_renorm(
                    probs, values.clone(), indices, thresholds, cumulative, tile
                )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                del actual, ordered
                print(
                    json.dumps(
                        dict(
                            check="shared_sort_exact",
                            batch=batch,
                            tile=tile,
                            passed=True,
                        )
                    ),
                    flush=True,
                )
                paired(
                    "key_sort",
                    lambda: keys.sort(-1).values,
                    lambda: op._sort_packed_keys(keys, tile),
                    args,
                    batch,
                    tile,
                )
                paired(
                    "filter_restore_including_clone",
                    lambda: baseline_restore(
                        probs, values.clone(), indices, thresholds, cumulative
                    ),
                    lambda: op.filter_and_renorm(
                        probs, values.clone(), indices, thresholds, cumulative, tile
                    ),
                    args,
                    batch,
                    tile,
                )
                paired(
                    "full_top_p",
                    lambda: full_top_p(probs, thresholds),
                    lambda: full_top_p(probs, thresholds, tile),
                    args,
                    batch,
                    tile,
                )
            del probs, thresholds, values, indices, cumulative, keys, expected
    print(
        "ALL_CHECKS_PASSED; positive paired_saved_ms means faster. NPU/TPOT validation is separate."
    )


if __name__ == "__main__":
    main()
