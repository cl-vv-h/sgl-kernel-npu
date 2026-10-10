# Fused candidate extraction and selected-row sampling experiment

Requires the companion SGLang fork branch `perf/npu-sampling-fused-selected-row`.
This kernel branch is `perf/sampling-fused-selected-row`. Both are based on the
community main fetched for this experiment; no other experiment is required.

The first probability sort and cumsum still process B*D rows. Filtering also
extracts each row's single-child candidate probability into small per-tile
partials, without a separate full-vocabulary scan. The denominator retains the
native sorted-value sum and clamp. Acceptance uses RN-divided candidate values
and the same coins/thresholds. After acceptance, only B rows of packed keys are
gathered and sorted. The selected restore produces the residual distribution,
eliminating the full B*D*V zero scratch. Final block sum/CDF kernels retain their
original arithmetic. This branch uses ordinary key sort, not SortV2.

It requires a single-child target-only tree (`--speculative-eagle-topk 1`),
FP32 contiguous probabilities, top-p enabled and RN division. Request-level
top-k filtering still runs normally before this path. Other configurations use
the existing SGLang path. The direct kernel API requires the caller to guarantee
no sibling alternatives and valid indices; it does not synchronize to inspect
tree contents. B is unrestricted: B=54,D=4 restores 54 rows instead of 216.

```bash
python -m pytest tests/python/sgl_kernel_npu/test_sampling_selected_row.py -q
python benchmark/bench_sampling_selected_row.py --batch-sizes 1 16 54 --vocab-size 248320 --draft-tokens 4
```

Checks compare exact candidate probabilities and all three output tensors with
shared sort inputs / fixed coins. They include permuted nodes, shortened paths,
partial depth, ties, zero mass, one-hot, top-p edges and acceptance thresholds.
The full-path benchmark measures top-p + target-only sampling, including
allocations; it is not a TPOT measurement. No NPU speedup is claimed in advance.

Server environment (restart required):

```bash
export SGL_KERNEL_NPU_SAMPLING_TRITON=1
export SGL_KERNEL_NPU_SAMPLING_TRITON_DIV=rn
export SGL_KERNEL_NPU_SAMPLING_SELECTED_ROW=1
```

SELECTED_ROW defaults to 0; use 0 as the V1+RN control on the same branch pair.
Keep temperature=1, top_p=0.95, concurrency, prompts, generation length and other
launch flags identical. Existing CLI flags are sufficient. Look for
`_selected_accept_kernel` and `_selected_restore_kernel` in profiling to confirm
activation. Changing only the kernel repository cannot activate this path:
SGLang must pass unfiltered probabilities and defer top-p restoration.

The implementation has not been compiled or benchmarked on NPU on the authoring
host. Run exact-output checks on your CANN/Triton version before full-model A/B.

## Ascend accept-kernel compilation compatibility

The initial implementation was reported to abort in `ttir_to_linalg` while
compiling `_selected_accept_kernel` (`TypeRange ... null types`). The stack
identifies the failing kernel but not the specific failing IR operation.
The compatibility revision keeps acceptance values as one-lane vectors and
statically unrolls depth, replacing nested runtime branches with masked
loads/stores and `where` updates. RN division, candidate probabilities and the
acceptance thresholds are unchanged. No new environment flag or SGLang update
is needed for this revision.

Check compilation in a single process before restarting the model service:

```bash
python benchmark/check_selected_accept.py --batch-sizes 1 54 --blocks 1 76 122
python -m pytest tests/python/sgl_kernel_npu/test_sampling_selected_row.py -q
python benchmark/bench_sampling_selected_row.py --batch-sizes 1 54
```

The CPU control-flow test executes the actual new and original kernel bodies
with a NumPy adapter. It checks semantics and masked accesses, not Ascend
lowering. The isolated NPU check above is still required to establish that the
compiler workaround succeeds on the target software stack.
