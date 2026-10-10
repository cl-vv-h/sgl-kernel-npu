# Values-only packed-key sort experiment

Based on community main plus V1 packed-key / Triton RN renormalization.
Only the second sort changes. The probability sort, cumsum, sorted reduction,
and RN division retain their original order. This branch is independent of
selected-row sampling and state-copy tiling.

Run these checks on the actual CANN / torch_npu stack **before** enabling:

```bash
python benchmark/bench_sampling_key_sort.py --batch-sizes 1 16 54 --vocab-size 248320 --full-top-p
python -m pytest tests/python/sgl_kernel_npu/test_speculative_probability_triton.py -q
```

The key probe requires bit-exact FP32 values, including large odd keys. Some
older SortV2 backends may not preserve these; an assertion failure disqualifies
this experiment on that stack. Do not relax the tolerance.

Enable for the server (restart required):

```bash
export SGL_KERNEL_NPU_SAMPLING_TRITON=1
export SGL_KERNEL_NPU_SAMPLING_TRITON_DIV=rn
export SGL_KERNEL_NPU_SAMPLING_VALUES_SORT=1
```

Set VALUES_SORT=0 for the V1+RN control on the same branch. Default is 0.
No SGLang change or new CLI flag is required. Compare full TPOT and acceptance
length with identical requests; key-sort microbenchmark savings are not TPOT.
NPU compilation, correctness and speed must be verified on the target host.
