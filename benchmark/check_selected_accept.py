"""Isolate Ascend compilation / equality of the selected-row accept kernel.

No model or full-vocabulary sort is needed. Test this before the full benchmark.
"""

import argparse
import json


def check(batch, blocks, draft=4, depth=None):
    import torch
    from sgl_kernel_npu.sample.selected_row import _selected_accept_kernel
    from sgl_kernel_npu.sample.tree_speculative_sampling_target_only import (
        _tree_target_only_accept_kernel,
    )

    depth = draft if depth is None else depth
    vocab = 8
    generator = torch.Generator().manual_seed(17)
    candidates = torch.randint(
        vocab, (batch, draft), generator=generator, dtype=torch.int64
    )
    index = torch.randperm(batch * draft, generator=generator).reshape(batch, draft)
    children = torch.full((batch, draft), -1, dtype=torch.int64)
    order = [0] + list(range(draft - 1, 0, -1))
    for parent, child in zip(order, order[1:]):
        children[:, parent] = child
    if draft > 1:
        children[::3, order[1]] = -1
    siblings = torch.full_like(children, -1)
    coins = torch.rand(batch, draft, generator=generator)
    coins[0].zero_()
    if batch > 1:
        coins[1].fill_(1)
    denominator = torch.rand(batch * draft, generator=generator) + 0.1
    numerator = torch.rand(batch * draft, generator=generator) * denominator
    partials = torch.zeros(batch * draft, blocks)
    for row in range(batch * draft):
        partials[row, row % blocks] = numerator[row]
    target = torch.zeros(batch, draft, vocab)
    for req in range(batch):
        for row in range(draft):
            child = children[req, row].item()
            if child >= 0:
                target[req, row, candidates[req, child]] = (
                    numerator[req * draft + row] / denominator[req * draft + row]
                )
    candidates, index, children, siblings, coins, partials, denominator, target = [
        t.npu()
        for t in (
            candidates,
            index,
            children,
            siblings,
            coins,
            partials,
            denominator,
            target,
        )
    ]

    def outputs():
        return [
            torch.zeros(batch * draft, dtype=torch.int32, device="npu"),
            torch.full((batch, depth), -1, dtype=torch.int32, device="npu"),
            torch.empty(batch, dtype=torch.int32, device="npu"),
            torch.empty((batch, 2), dtype=torch.int64, device="npu"),
        ]

    expected, actual = outputs(), outputs()
    scratch = torch.zeros_like(target)
    rejected = torch.empty(batch, dtype=torch.int64, device="npu")
    print(
        json.dumps(
            dict(
                stage="original_accept",
                batch=batch,
                draft=draft,
                depth=depth,
                blocks=blocks,
            )
        ),
        flush=True,
    )
    _tree_target_only_accept_kernel[(batch,)](
        *expected[:3],
        candidates,
        index,
        children,
        siblings,
        coins,
        target,
        scratch,
        expected[3],
        1.0,
        1.0,
        draft,
        depth,
        vocab,
    )
    torch.npu.synchronize()
    print(
        json.dumps(
            dict(
                stage="selected_accept",
                batch=batch,
                draft=draft,
                depth=depth,
                blocks=blocks,
            )
        ),
        flush=True,
    )
    _selected_accept_kernel[(batch,)](
        *actual[:3],
        candidates,
        index,
        children,
        coins,
        partials,
        denominator,
        actual[3],
        rejected,
        1.0,
        1.0,
        draft,
        depth,
        blocks,
        1 << (blocks - 1).bit_length(),
    )
    torch.npu.synchronize()
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, rtol=0, atol=0)
    target, scratch, rejected, metadata = [
        t.cpu() for t in (target, scratch, rejected, actual[3])
    ]
    for req in range(batch):
        row = int(metadata[req, 0])
        residual = target[req, row].clone()
        if int(rejected[req]) >= 0:
            residual[int(rejected[req])] = 0
        torch.testing.assert_close(
            residual,
            (target[req, row] - scratch[req, row]).clamp_min(0),
            rtol=0,
            atol=0,
        )
    print("EXACT_ACCEPT_PASSED", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 54])
    parser.add_argument("--blocks", type=int, nargs="+", default=[1, 76, 122])
    args = parser.parse_args()
    if min(args.batch_sizes + args.blocks) < 1:
        parser.error("batch sizes and blocks must be positive")
    import torch
    import torch_npu
    import triton

    print(
        json.dumps(
            dict(
                torch=torch.__version__,
                torch_npu=torch_npu.__version__,
                triton=triton.__version__,
            )
        ),
        flush=True,
    )
    with torch.inference_mode():
        for batch in args.batch_sizes:
            for blocks in args.blocks:
                check(batch, blocks)
        check(1, 1, draft=1)
        check(4, 122, depth=3)


if __name__ == "__main__":
    main()
