# Mamba state-copy tiling experiment

Independent of values-only sort and selected-row sampling. Includes V1+RN as a
common sampling baseline. Does not modify convolution rollback or KDA copying.

The original program owns one request and serially loops over every layer and
head block. The new persistent grid distributes disjoint request/layer/state
subtiles across vector cores, with 64-bit addressing and explicit source and
destination strides. It skips negative steps and requires unique active
destination indices, as does the existing parallel copy. Shared source/destination
storage retains the original implementation.

```bash
python -m pytest tests/python/sgl_kernel_npu/test_mamba_state_copy_tiled.py -q
python benchmark/bench_mamba_state_copy.py --batch-sizes 1 16 54 --layers 8 --heads 4 --dim-v 128 --dim-k 128
```

Set layers/heads/V/K/dtype to the actual local-rank cache dimensions before
using a microbenchmark to select the block size. Defaults limit memory use and
are not a claim about your model's state shape. The benchmark checks exact state
contents, including strided tensors, negative steps and unused destination slots.

Server settings (restart for each A/B run):

```bash
export SGL_KERNEL_NPU_SAMPLING_TRITON=1
export SGL_KERNEL_NPU_SAMPLING_TRITON_DIV=rn
export SGL_KERNEL_NPU_STATE_COPY_TILED=1
export SGL_KERNEL_NPU_STATE_COPY_BLOCK=4096
```

TILED defaults to 0, BLOCK to 4096. Supported blocks: 1024, 2048, 4096, 8192.
Set TILED=0 for the original copy with the same V1+RN sampling baseline.
No SGLang changes or new CLI flags are required. Restart after changing flags
because captured graphs retain the kernel chosen during capture. NPU compilation,
state equivalence and full-model TPOT remain target-host validation requirements.
