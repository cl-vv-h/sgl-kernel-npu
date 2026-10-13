# Independent boundary restoration and scan fusion experiments

Base: `fcf14ce` on `perf/fused-sampling-renorm` (FP32 V1+RN). This experimental
branch does not alter the stable PR or SGLang. Both switches default to zero.
All native probability sorting, probability tensors, denominators and RN
division remain FP32. This is not the INT32-key experiment or AIR implementation.

## Startup switches

| `SGL_KERNEL_NPU_SAMPLING_BOUNDARY_RESTORE` | `SGL_KERNEL_NPU_SAMPLING_SCAN_FUSION` | Path |
| --- | --- | --- |
| 0 | 0 | Original V1+RN probability sort / native cumsum / packed-key sort / RN restore |
| 1 | 0 | Native cumsum, fused boundary statistics, contiguous restore and bounded correction |
| 0 | 1 | Tiled scan/filter fusion, original FP32 packed-key sort and RN restore |
| 1 | 1 | Tiled scan/filter fusion followed by boundary restoration |

Only the string `1` enables a switch. Values are read once when `probability.py`
is imported, not during each operator invocation. Set them before starting the
service; restart after changing them. Existing `SAMPLING_TRITON`, `TRITON_DIV`,
AIR, compact and INT32 switches are not used by this branch. Division is fixed RN.
Top-k can use boundary restoration; the scan switch only affects top-p.

## Boundary restoration

The previous boundary implementation separately scanned the matrix to produce
metadata and divided the output after restoration. This implementation collects
metadata during filtering and fuses RN division into both restoration kernels.
It does not allocate full-vocabulary keys, sort them, or clone the original
sorted probability matrix in the production path.

The native descending sort provides values and a token-ID permutation. For each
block, the filter records four FP32 values: minimum positive retained probability,
maximum positive removed probability, and the original first/last probabilities.
Filtering still overwrites sorted values, so the unchanged native sum computes
the denominator from the same shape and layout as V1.

Let `a` be the minimum positive retained probability and `b` the maximum positive
removed probability. A contiguous output kernel initially keeps original values
strictly greater than `a`. Every potentially incorrect positive decision lies
in `[a, max(a, b)]`: retained values cannot be smaller than `a`, and removed
values cannot exceed `b`. A small reduction finds the first/last sorted blocks
intersecting this interval. A correction kernel writes the actual filtered
values through the original native sort indices only for those blocks.

Correcting entire intersecting blocks avoids keeping a full original sorted copy
for another binary search. It preserves native token identities inside ties and
also covers non-prefix decisions due to floating-point cutoff rounding. Original
zero probabilities remain zero. All-zero rows skip correction entirely. No
assumption about stable token-ID ordering in the native sort is made.

Contiguous initialization and indexed correction are separate kernel launches
on the caller's stream. Correction has one program owner per row; programs do
not compete to write different indexed blocks within that row. Scratch output
rows are padded to 32-byte alignment. Production vocabularies 154880/248320 need
no extra output copy; nonaligned tail shapes return a contiguous copy.

Constraints: finite, nonnegative FP32 probabilities; native descending sorted
values and permutation; same existing vocabulary limit `1 <= V <= 2**23`.
Large boundary groups, including uniform distributions, may require correcting
most of a row and may regress. There is no unmeasured automatic fallback. The
benchmark reports correction block counts and fractions to explain such cases.
Indexed writes still need device compilation, correctness and graph validation;
CPU emulation does not validate the compiler's memory ordering.

## Scan fusion

Instead of writing/reading a full `[rows, vocab]` cumulative matrix, one kernel
computes FP32 tile totals. A small native cumsum on `[rows, tiles]` provides each
tile's preceding carry. The filter computes `tl.cumsum(values, 0) + carry`, then
uses the existing `prefix - value > top_p` predicate. It either emits FP32 keys
or boundary statistics, according to the other switch.

For `216 x 248320` and block 2048, the full cumulative matrix is about 204.61 MiB.
Tile totals and carries are each about 102.94 KiB. There is still an extra full
read for tile sums; reduced temporary storage is not proof of lower latency.

**Precision limitation:** tile sums and local scans change FP32 addition order.
RN division does not correct prefix rounding. Near a cutoff, the retained set
may differ from native cumsum. Boundary-only aims at exact V1 equivalence; scan
and combined paths are experimental and require separate retained-set and model
accuracy/accept-length validation. The benchmark never silently relaxes boundary
exactness or restoration exactness to make a scan experiment pass.

New kernels keep rows as a runtime argument (`do_not_specialize`), with vocab
and tile sizes specialized. Graph replay still uses the fixed shape of its
capture. No host `.item()`, dynamic `nonzero`, stream switches or explicit device
synchronization is added to the operator path.

## Benchmark without installing

Run from this checkout/archive with the existing NPU torch_npu/Triton environment:

```bash
python benchmark/bench_sampling_boundary_scan.py \
  --batch-sizes 20 54 --vocab-size 248320 --draft-tokens 4 \
  --modes boundary scan both --iterations 30 --breakdown
```

Start with `--batch-sizes 1 --vocab-size 2051 --iterations 5` for a compile and
correctness smoke check. Add `--graph` for capture/replay timing. Use
`--modes boundary` to isolate the exact native-cumsum candidate, or `scan`/`both`
to test the other combinations. Temperature defaults to 0.6, top-p to 0.95.
Both are probability preparation inputs outside timing.

The script loads the candidate file directly by path and does not overwrite the
installed package or the server. It explicitly selects helpers, so imported
startup switches do not affect A/B dispatch. It compares every candidate to the
same native probability permutation, checks restoration against a scatter oracle
using its actual filtered values, and prints native-V1 probability/keep mismatches.
Native-scan boundary differences always abort. `--require-v1-exact` also aborts
for scan-rounding differences. Without it, scan timing proceeds for diagnosis,
but a result with differences is not precision-approved.

`full_top_p` includes the initial probability sort, cumsum or tiled scan, filter,
sum, key sorting or correction, and RN restoration. `renorm_including_clone`
excludes the probability sort and includes identical input-clone costs on both
sides. Timings alternate A/B order and synchronize both endpoints. Positive
`paired_saved_ms` means improvement. Correctness transfers/statistics are outside
the timed sections. Fixed finite test cases cannot prove universal equivalence.

## Device tests

After updating the installed Python source and restarting:

```bash
python -m pytest tests/python/sgl_kernel_npu/test_speculative_probability.py -q
```

The fixture resets both switches, then parameterized cases explicitly cover
baseline/boundary shared-sort correctness, all public switch combinations,
scan-packed versus scan-boundary restoration, artificial non-prefix selection,
and changed-input graph replay for all four combinations. The scan tests compare
against the same tiled-scan selection, not a blanket native-cumsum equivalence
claim. Service acceptance length, AIME accuracy and TPOT require separate tests.

This machine has no NPU; source-body CPU checks, syntax and formatting checks
are local evidence only. No NPU compilation or performance gain is claimed.
