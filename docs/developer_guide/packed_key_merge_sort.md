# Values-only packed-key merge experiment

Branch: `perf/packed-key-merge-sort`. Baseline: `fcf14ce` (fixed V1+RN).
The existing PR branch is unchanged. This candidate replaces only the second
encoded-key sorting/restoration pipeline. It is not a measured speedup yet.

## Operator change

- The first probability sort, inclusive native cumsum, `prefix-current > top_p`
  predicate, native masked sum/clamp, and RN division remain unchanged.
- Filtering encodes the same exact FP32 `2*token_id+keep` keys and sorts each
  2048-element tile on chip, using Triton-Ascend `extension.sort`.
- Merge-path partitions consecutive output tiles across two sorted runs.
  Bounded suffixes are loaded, locally sorted, and stored consecutively.
- Seven merge levels cover vocab=248320 at tile=2048. The final level restores
  probabilities and performs `div_rn` directly, without materializing the final
  ordered-key matrix or launching the separate restoration kernel.
- No full-vocabulary output indices, INT64 output-index Cast, token-indexed
  global scatter, host readback, atomics, custom communication or environment
  dispatch is introduced. Intermediate buffers belong to each invocation.

This is a values-only output algorithm; the compiler's internal implementation
of local sort may still use local temporary indices. It is not a general sort:
keys must be finite and unique, as the packed token permutation guarantees.
FP32 encoding/odd keep bits must survive local sorting exactly.

The implementation adds merge passes and scalar partition probes. Those costs
can outweigh removal of native indices/Cast. Judge the complete chain, not just
memory savings. First probability sorting and cumsum are still native; this
commit does not implement the separate prefix-scan fusion candidate.

The Ascend extension is part of the existing Triton-Ascend dependency; there is
no new pip package or C++ build target. The installed compiler must support
`triton.language.extra.cann.extension.sort`:
https://triton-ascend.readthedocs.io/en/latest/python-api/generated/triton.language.extra.cann.extension.sort.html

## Benchmark before installation

From this checkout, with the existing NPU torch/triton/sgl_kernel_npu environment:

```bash
python benchmark/bench_packed_key_merge.py \
  --batch-sizes 24 54 --vocab-size 248320 --draft-tokens 4 \
  --tiles 2048 --iterations 30
```

The benchmark loads this checkout's probability.py directly and does not replace
site-packages. Metadata prints the actual source path. Baseline reproduces
fcf14ce V1+RN using the shared filtering body with local key sorting disabled,
native key sort and the original restoration kernel. The production path has
one fixed implementation and no environment switches.

The correctness gate shares the first sort permutation to avoid tie ambiguity,
requires exact probabilities and checks ordered keys. Timing alternates A/B order
and synchronizes both endpoints. `paired_saved_ms > 0` means faster.
The restore diagnostic includes the same values clone on both sides; full_top_p
has no diagnostic clone. These are operator measurements, not TPOT results.

If eager correctness passes, test graph timing:

```bash
python benchmark/bench_packed_key_merge.py \
  --batch-sizes 54 --vocab-size 248320 --draft-tokens 4 \
  --tiles 2048 --iterations 30 --graph
```

For compiler/UB errors or tile tuning, the same benchmark accepts
`--tiles 512 1024 2048`. This changes only its explicit test parameter; service
uses the fixed 2048 tile. A smaller tile adds merge levels. No silent fallback is
used after a compilation error. Both vocab=154880 and 248320 are supported.

## NPU tests and service

Install this branch using your existing kernel package workflow, then run:

```bash
pytest tests/python/sgl_kernel_npu/test_speculative_probability.py -q
```

The tests cover shared-sort exact renormalization, ties/zero/tiny probabilities,
216-row large vocabularies, scalar/strided thresholds, high-bit odd FP32 keys,
ordered/reversed/random permutations and graph replay with changed inputs.
CPU-only hosts skip the NPU suite. Local CPU kernel-body diagnostics validate
algorithm/indexing only; they do not validate NPU compilation, memory ordering,
compiler precision or speed.

Restart workers after installation. No new launch argument or environment
variable is required, and old sampling switches do not select this path.
Compare end-to-end TPOT/throughput against the unchanged V1+RN PR branch after
operator correctness and the complete chain show a positive measured benefit.
