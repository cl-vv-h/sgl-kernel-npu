"""Exact-output checks and end-to-end top-p + tree sampling A/B benchmark."""

import argparse
import json
import statistics
import time


def make_inputs(batch, draft, vocab, device, case="random", short=False):
    import torch

    generator = torch.Generator().manual_seed(17)
    probs = torch.softmax(
        torch.randn(batch * draft, vocab, generator=generator) * 4, -1
    )
    if case == "ties":
        probs.fill_(1.0 / vocab)
    elif case == "zeros":
        probs.zero_()
    elif case == "onehot":
        probs.zero_()
        probs[:, -1] = 1
    # Permuted node numbering tests that row selection follows next_token,
    # rather than assuming accepted length == final node number.
    order = [0] + list(range(draft - 1, 0, -1))
    children = torch.full((batch, draft), -1, dtype=torch.int64)
    candidates = torch.randint(vocab, (batch, draft), generator=generator)
    for parent, child in zip(order, order[1:]):
        children[:, parent] = child
        candidates[:, child] = probs.view(batch, draft, vocab)[:, parent].argmax(-1)
    if short and draft > 1:
        children[:, order[1]] = -1
    coins = torch.rand(batch, draft, generator=generator)
    coins[0].zero_()  # all accept, including probability zero as in baseline
    if batch > 1:
        coins[1].fill_(1)  # force rejection unless threshold_single applies
    final = torch.rand(batch, generator=generator)
    final[0] = 0
    if batch > 1:
        final[1] = 1
    return {
        "target_probs": probs.to(device),
        "candidates": candidates.to(device),
        "retrive_index": torch.arange(
            batch * draft, dtype=torch.int64, device=device
        ).reshape(batch, draft),
        "retrive_next_token": children.to(device),
        "retrive_next_sibling": torch.full(
            (batch, draft), -1, dtype=torch.int64, device=device
        ),
        "uniform_samples": coins.to(device),
        "uniform_samples_for_final_sampling": final.to(device),
    }


def outputs(batch, draft, device, depth=None):
    import torch

    return dict(
        predicts=torch.zeros(batch * draft, dtype=torch.int32, device=device),
        accept_index=torch.full(
            (batch, depth or draft), -1, dtype=torch.int32, device=device
        ),
        accept_token_num=torch.empty(batch, dtype=torch.int32, device=device),
    )


def check_shared_sort(
    vocab=2051,
    draft=4,
    case="random",
    short=False,
    threshold_single=1.0,
    threshold_acc=1.0,
    depth=None,
):
    import torch
    from sgl_kernel_npu.sample.probability_triton import filter_and_renorm
    from sgl_kernel_npu.sample.selected_row import _finish_selected, _prepare_selected
    from sgl_kernel_npu.sample.tree_speculative_sampling_target_only import (
        tree_speculative_sampling_target_only,
    )

    batch = 5
    data = make_inputs(batch, draft, vocab, "npu", case, short)
    probs = data["target_probs"]
    top_ps = torch.tensor([0.0, 0.25, 0.5, 0.95, 1.0], device="npu").repeat_interleave(
        draft
    )
    values, indices = probs.sort(-1, descending=True)
    baseline = filter_and_renorm(
        probs, values.clone(), indices, top_ps, values.cumsum(-1), division="rn"
    )
    keys, partials, denom = _prepare_selected(
        probs,
        top_ps,
        data["candidates"],
        data["retrive_next_token"],
        (values.clone(), indices),
    )
    # Candidate probabilities must match the full RN tensor exactly, including
    # ties at the top-p boundary and zero mass.
    child = data["retrive_next_token"]
    ids = data["candidates"].gather(1, child.clamp_min(0)).reshape(-1, 1)
    wanted = baseline.gather(1, ids).reshape(batch, draft).masked_fill(child < 0, 0)
    actual = (partials.max(-1).values / denom.flatten()).reshape(batch, draft)
    torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
    expected, actual_outputs = outputs(batch, draft, "npu", depth), outputs(
        batch, draft, "npu", depth
    )
    kwargs = {k: v for k, v in data.items() if k != "target_probs"}
    tree_speculative_sampling_target_only(
        **expected,
        **kwargs,
        target_probs=baseline.reshape(batch, draft, vocab),
        draft_probs=torch.empty_like(baseline.reshape(batch, draft, vocab)),
        threshold_single=threshold_single,
        threshold_acc=threshold_acc
    )
    kwargs.pop("retrive_next_sibling")
    _finish_selected(
        probs,
        keys,
        partials,
        denom,
        **actual_outputs,
        **kwargs,
        threshold_single=threshold_single,
        threshold_acc=threshold_acc
    )
    for name in expected:
        torch.testing.assert_close(actual_outputs[name], expected[name], rtol=0, atol=0)


def benchmark(batch, draft, vocab, warmup, iterations):
    import torch
    from sgl_kernel_npu.sample.probability import top_p_renorm_prob
    from sgl_kernel_npu.sample.selected_row import top_p_tree_sampling_selected_row
    from sgl_kernel_npu.sample.tree_speculative_sampling_target_only import (
        tree_speculative_sampling_target_only,
    )

    data = make_inputs(batch, draft, vocab, "npu")
    original = data["target_probs"]
    kwargs = {k: v for k, v in data.items() if k != "target_probs"}
    top_ps = torch.full((batch * draft,), 0.95, device="npu")

    def baseline():
        result = outputs(batch, draft, "npu")
        probs = top_p_renorm_prob(original, top_ps).reshape(batch, draft, vocab)
        tree_speculative_sampling_target_only(
            **result, **kwargs, target_probs=probs, draft_probs=torch.zeros_like(probs)
        )
        return result

    def selected():
        result = outputs(batch, draft, "npu")
        top_p_tree_sampling_selected_row(**result, **data, top_ps=top_ps)
        return result

    expected, actual = baseline(), selected()
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
    paths = {"v1_rn": baseline, "selected_row": selected}
    for _ in range(warmup):
        for fn in paths.values():
            fn()
    torch.npu.synchronize()
    durations = {name: [] for name in paths}
    for i in range(iterations):
        for name in list(paths)[:: 1 if i % 2 == 0 else -1]:
            torch.npu.synchronize()
            start = time.perf_counter()
            result = paths[name]()
            torch.npu.synchronize()
            durations[name].append((time.perf_counter() - start) * 1000)
            del result
    medians = {name: statistics.median(times) for name, times in durations.items()}
    print(
        json.dumps(
            dict(
                batch=batch,
                draft=draft,
                vocab=vocab,
                check="EXACT_OUTPUTS_PASSED",
                median_ms=medians,
                saved_ms=medians["v1_rn"] - medians["selected_row"],
                scope="top-p + target-only sampling, not TPOT",
            )
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 16, 54])
    parser.add_argument("--draft-tokens", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=248320)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    if (
        min(args.batch_sizes) < 1
        or args.draft_tokens < 1
        or not 1 <= args.vocab_size <= 2**23
        or args.warmup < 0
        or args.iterations < 1
    ):
        parser.error("invalid shape or iteration count")
    import os

    import torch
    import torch_npu

    os.environ["SGL_KERNEL_NPU_SAMPLING_TRITON"] = "1"
    os.environ["SGL_KERNEL_NPU_SAMPLING_TRITON_DIV"] = "rn"
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
    with torch.inference_mode():
        for case in ["random", "ties", "zeros", "onehot"]:
            check_shared_sort(case=case)
        for batch in args.batch_sizes:
            benchmark(
                batch, args.draft_tokens, args.vocab_size, args.warmup, args.iterations
            )


if __name__ == "__main__":
    main()
