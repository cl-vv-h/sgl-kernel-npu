# Sampling scan/filter fusion experiment

Branch: `perf/sampling-scan-filter-fusion`. Base: `fcf14ce` on
`perf/fused-sampling-renorm`. The stable PR branch is unchanged.

## Direction after the merge regression

The user reported that `perf/packed-key-merge-sort` reduced throughput from
1300+ to 300+ tokens/s. That branch should be rolled back to V1+RN. This report
is end-to-end evidence of regression, not a measured per-kernel breakdown.
Multi-level full-vocabulary passes, local sorting and scalar partition probes
are plausible overheads; no hardware trace of that run is available locally.

Consequently this service candidate uses neither probability merge nor key
merge. Both descending probability sort and ascending packed-key sort remain
native. Native sorting preserves the original identities inside tied values.
The branch does not modify SGLang or introduce environment dispatch.

## Service change

V1: full native cumsum -> filter/encode -> native sum -> native key sort -> RN restore.

Candidate: tile sums -> small carry cumsum -> fused tile scan/filter/encode ->
native sum -> native key sort -> RN restore.

- `_probability_tile_totals_kernel` computes one FP32 sum per tile.
- A native cumsum over `[rows, tiles]` generates carry values.
- `_scan_filter_encode_kernel` scans each sorted tile and adds the preceding
  carry, applies `prefix-current > top_p`, writes masked values and packed keys.
- The original native masked-value sum/clamp, FP32 encoding and `tl.div_rn`
  restoration are retained. Top-k keeps the V1 implementation.

At rows=216, vocab=248320, tile=2048, the full FP32 cumulative array is about
204.61 MiB; each carry/totals array is about 102.94 KiB. The new totals pass
still reads the full sorted array. Eliminating a full cumulative write/read
is not proof that the replacement kernels run faster.

### Precision boundary

This is an experimental implementation, not a bitwise-equivalent replacement
claim. Tile sums and local scans change FP32 addition order. RN only preserves
the division operation; it does not eliminate prefix rounding differences.
A threshold within an ULP of a cutoff can change the retained token set.
No approximation to temperature/top-p or probability dtype is introduced.

The benchmark prints mismatched keys, retained sets, probabilities, maximum
absolute error and a categorical-sampling diagnostic. The diagnostic is not
an MTP acceptance-length or model-accuracy test. `--require-v1-exact` stops
before timing when any compared key/probability differs. Finite test cases
cannot establish universal bitwise equivalence. Do not label passing binary
probability tests as proof of equivalence on arbitrary FP32 inputs.

## INT32 first-sort diagnostic

`_sort_probabilities_int32` implements bounded FP32 probability/INT32 token-ID
sorting and merge-path partitioning. Initial tiles use a pairwise sorting
network; merged suffixes are loaded into UB and output ranks are located there.
All full-vocabulary addresses use INT64 arithmetic. It emits INT32 IDs directly,
without first creating INT64 output and casting it down.

It is diagnostic-only and never called by the public sampling functions. Equal
probabilities are ordered by increasing token ID, which can differ from native
unstable sorting. The benchmark records identity differences and the helper
is not combined with the service candidate. After the packed-key regression,
no end-to-end benefit is assumed for this related multi-level implementation.

The documented `aclnnSort` output index type is INT64; a Python `.int()` after
native sorting does not remove that allocation or internal Cast. The current
op-plugin wrapper also allocates `at::kLong` output. An INT32 native output
cannot be promised without a supported backend interface and device validation:

- https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha001/apiref/aolapi/context/aclnnSort.md
- https://github.com/Ascend/op-plugin/blob/master/op_plugin/ops/opapi/SortKernelNpuOpApi.cpp
- https://triton-ascend.readthedocs.io/en/latest/python-api/generated/triton.language.cumsum.html

## Benchmark without installation

Run from this checkout in the existing NPU torch/triton/sgl_kernel_npu environment:

```bash
python benchmark/bench_sampling_scan_fusion.py \
  --batch-sizes 10 20 33 54 --vocab-size 248320 --draft-tokens 4 \
  --tiles 2048 --iterations 30
```

The benchmark loads this checkout's probability.py directly. It does not modify
site-packages or the running server. Metadata prints source path/hash and fixed
RN division. `scan_filter_including_clone` isolates preparation; both sides
clone the same sorted values. `full_top_p` includes both native sorts and all
normalization/restoration costs. A/B order alternates and both endpoints are
synchronized. Positive `paired_saved_ms` means faster. Graph timing is available
with `--graph`; tile comparisons use `--tiles 512 1024 2048`.

For strict V1 comparison, add `--require-v1-exact`. Boundary tests are deliberately
adversarial, so a failure can reveal the documented arithmetic difference.
Without this option, timing can proceed for diagnosis even when comparisons
are not exact; those timings do not make the candidate precision-approved.

The default source-load benchmark does not change the installed package. Use
`--modes scan` for scan-only testing. `--temperature` defaults to 0.6, matching
the three CI cases; it affects synthetic probability setup outside timing.
The local batch sizes 10/20/33/54 correspond to total concurrency 40 with DP4,
160 with DP8, 132 with DP4 and 432 with DP8. Draft tokens remain four.
New kernels keep ROWS at runtime (`do_not_specialize`) while vocab/tile stay
compile-time constants; dynamic row counts do not create one binary per batch.
Graph capture/replay still requires the fixed input shape used by that graph.

To inspect only the first-sort prototype, run an independent comparison:

```bash
python benchmark/bench_sampling_scan_fusion.py \
  --batch-sizes 20 54 --vocab-size 248320 --draft-tokens 4 \
  --modes sort-int32 --probability-tile 512 --iterations 30
```

`diagnostic_probability_sort` isolates sorting, while
`diagnostic_sort_full_top_p` compares the entire pipeline, retaining native
cumsum and key sort. Probability, keep-set and categorical mismatch counts
are printed independently of timing. `--modes scan sort-int32` runs both
candidates separately; they are never stacked. The older
`--diagnose-probability-merge` flag remains an alias for adding the diagnostic.
For strict output comparison, add `--require-v1-exact`.

It remains inactive in service regardless of benchmark flags. Do not deploy
it based solely on the removal of the native index Cast.

## NPU tests and service

After installing this branch through your existing package workflow:

```bash
pytest tests/python/sgl_kernel_npu/test_speculative_probability.py -q
```

The suite retains strict native-reference tests, including large-vocabulary
216-row graph replay. They can expose rounding incompatibility and must not be
silently relaxed. New tests cover INT32 pair-sort correctness with an explicit
stable tie oracle, native tie mismatch reporting, exact dyadic scan inputs and
changed-input graph replay. To test only the service candidate, use
`-k 'not probability_sort_int32'`.

Local CPU kernel-body tests do not verify NPU compilation, memory ordering,
compiler precision, throughput or TPOT. The NPU suite is skipped on CPU hosts.
There are no new packages or launch/environment parameters. The branch's
public top-p function defaults to fused scan/filter at tile=2048 after install
and restart; both native sorts and RN remain. Keep V1+RN as the service baseline
until correctness and whole-chain performance have been checked.
