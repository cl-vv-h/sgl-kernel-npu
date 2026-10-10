#!/usr/bin/env python3
"""NPU AIR vs V1+RN: compile smoke, correctness, eager/graph and stage timing."""

import argparse
import json
import os
import statistics
import time

import numpy as np
import torch
import torch_npu  # Registers torch.npu before importing the kernel package.
import triton
from sampling_air_reference import validate
from sgl_kernel_npu.sample import probability, probability_air


def emit(**record):
    print(json.dumps(record, ensure_ascii=False), flush=True)


def make_input(rows, vocab, case, device):
    generator = torch.Generator().manual_seed(20261010)
    if case in ("softmax", "peaked"):
        logits = torch.randn(rows, vocab, generator=generator)
        cpu = torch.softmax(logits * (4 if case == "peaked" else 1), dim=-1)
    elif case == "uniform":
        cpu = torch.full((rows, vocab), 1.0 / vocab)
    else:
        cpu = torch.randint(0, 16, (rows, vocab), generator=generator).float()
        cpu /= cpu.sum(-1, keepdim=True).clamp_min(1e-20)
    return cpu.numpy(), cpu.to(device)


def stage_hook():
    starts = {}

    def hook(name, begin):
        torch.npu.synchronize()
        if begin:
            emit(stage=name, state="begin")
            starts[name] = time.perf_counter()
        else:
            emit(
                stage=name,
                state="end",
                synchronized_wall_ms=1000 * (time.perf_counter() - starts[name]),
            )

    return hook


def graph_callable(fn):
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            expected = fn()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream, auto_dispatch_capture=True):
        output = fn()
    torch.npu.synchronize()
    for _ in range(2):
        output.fill_(float("nan"))
        torch.npu.synchronize()
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(output, expected, rtol=3e-6, atol=1e-8)

    def replay():
        graph.replay()
        return output

    return replay


def timing_pair(functions, warmup, iterations):
    for _ in range(warmup):
        for fn in functions.values():
            fn()
    torch.npu.synchronize()
    samples = {name: [] for name in functions}
    entries = list(functions.items())
    for iteration in range(iterations):
        # Alternate order to reduce systematic temperature/order effects.
        for name, fn in entries[:: 1 if iteration % 2 == 0 else -1]:
            torch.npu.synchronize()
            start = time.perf_counter()
            output = fn()
            torch.npu.synchronize()
            samples[name].append(1000 * (time.perf_counter() - start))
            del output
    return {
        name: {"median_ms": statistics.median(times), "min_ms": min(times)}
        for name, times in samples.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 16, 54])
    parser.add_argument("--vocab-size", type=int, default=248320)
    parser.add_argument("--draft-tokens", type=int, default=4)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--blocks", type=int, nargs="+", choices=[256, 512, 1024], default=[512]
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=["softmax", "peaked", "uniform", "ties"],
        default=["softmax"],
    )
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Compile tiny and requested shapes; skip latency loops",
    )
    parser.add_argument(
        "--stage-times",
        action="store_true",
        help="Print each launch before/after synchronized diagnostic execution",
    )
    parser.add_argument(
        "--graph",
        action="store_true",
        help="Also compare graph replay; fail if graph capture/replay fails",
    )
    args = parser.parse_args()
    if (
        min(args.batch_sizes) < 1
        or min(args.vocab_size, args.draft_tokens, args.iterations) < 1
        or args.warmup < 0
    ):
        parser.error("shapes/iterations must be positive and warmup nonnegative")
    if not 0 <= args.top_p <= 1:
        parser.error("top-p must be in [0, 1]")
    torch.npu.set_device(args.device)
    os.environ["SGL_KERNEL_NPU_SAMPLING_TRITON"] = "1"
    os.environ["SGL_KERNEL_NPU_SAMPLING_TRITON_DIV"] = "rn"
    os.environ["SGL_KERNEL_NPU_SAMPLING_AIR"] = "0"
    emit(
        torch=torch.__version__,
        torch_npu=torch_npu.__version__,
        triton=triton.__version__,
        device=torch.npu.get_device_name(),
        probability_source=probability.__file__,
        air_source=probability_air.__file__,
        args=vars(args),
        note="V1 uses public dispatch with AIR=0; AIR calls the experimental function directly. Timing is synchronized wall time.",
    )
    hook = stage_hook() if args.stage_times or args.smoke_only else None

    # Early compile/edge failures are easy to locate before the large allocation.
    edge = torch.tensor(
        [
            [0.5, 0.25, 0.25, 0],
            [0.25, 0.25, 0.25, 0.25],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    ps_cpu = np.array([0.5, 0.0, 0.95, 1.0], np.float32)
    for block in args.blocks:
        emit(check="edge_compile", block=block)
        result = probability_air.air_top_p_renorm(
            edge.to(args.device),
            torch.from_numpy(ps_cpu).to(args.device),
            block,
            stage_hook=hook,
        )
        emit(
            check="edges",
            block=block,
            **validate(edge.numpy(), ps_cpu, result.cpu().numpy())
        )

    for batch in args.batch_sizes:
        rows = batch * args.draft_tokens
        for case in args.cases:
            cpu, probs = make_input(rows, args.vocab_size, case, args.device)
            ps = torch.full(
                (rows,), args.top_p, device=args.device, dtype=torch.float32
            )
            cpu_ps = np.full(rows, args.top_p, np.float32)
            baseline = lambda: probability.top_p_renorm_prob(probs, ps)
            emit(check="v1_compile", batch=batch, rows=rows, case=case)
            v1 = baseline().cpu().numpy()
            for block in args.blocks:
                context = dict(
                    batch=batch,
                    rows=rows,
                    vocab=args.vocab_size,
                    case=case,
                    block=block,
                )
                emit(check="air_compile", **context)
                air_fn = lambda: probability_air.air_top_p_renorm(probs, ps, block)
                # Diagnostic runs include JIT compilation; not headline latency.
                actual = (
                    probability_air.air_top_p_renorm(probs, ps, block, stage_hook=hook)
                    .cpu()
                    .numpy()
                )
                metrics = validate(cpu, cpu_ps, actual)
                tv_v1 = 0.5 * np.abs(actual.astype(np.float64) - v1).sum(axis=1)
                emit(
                    check="correctness",
                    **context,
                    **metrics,
                    max_tv_vs_v1=float(tv_v1.max()),
                    support_diff_vs_v1=int(np.count_nonzero((actual > 0) != (v1 > 0)))
                )
                # Exercise service dispatch too; no graph captures are active yet.
                os.environ["SGL_KERNEL_NPU_SAMPLING_AIR"] = "1"
                os.environ["SGL_KERNEL_NPU_SAMPLING_AIR_BLOCK"] = str(block)
                dispatched = probability.top_p_renorm_prob(probs, ps).cpu().numpy()
                os.environ["SGL_KERNEL_NPU_SAMPLING_AIR"] = "0"
                np.testing.assert_array_equal(dispatched, actual)
                if args.smoke_only:
                    continue
                functions = {"v1_rn": baseline, "air_rn": air_fn}
                emit(
                    mode="eager",
                    **context,
                    timings=timing_pair(functions, args.warmup, args.iterations)
                )
                if args.stage_times:
                    emit(check="warm_stage_times", **context)
                    probability_air.air_top_p_renorm(
                        probs, ps, block, stage_hook=stage_hook()
                    )
                if args.graph:
                    emit(check="graph_capture", **context)
                    graphs = {
                        name: graph_callable(fn) for name, fn in functions.items()
                    }
                    emit(
                        mode="graph",
                        **context,
                        timings=timing_pair(graphs, args.warmup, args.iterations)
                    )
                    del graphs
    emit(
        status="passed",
        note="Microbenchmark only; measure TPOT, acceptance length and output quality separately.",
    )


if __name__ == "__main__":
    main()
