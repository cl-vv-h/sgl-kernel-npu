# Experimental AIR Top-P on Ascend

本分支将 FlashInfer AIR Top-P 的**按概率质量选择 radix 边界**算法移植到
Ascend Triton，供独立 A/B 验证。不是 CUDA 二进制复用，也不承诺 GPU 的收益
能直接转移到 NPU。只修改 kernel 仓，SGLang 调用接口不变。

## 基线与范围

- 社区 main：`36e8a8e399d15585279d5b7ef014396632fbea8f`。
- 分支：`perf/sampling-air-top-p`。
- 在 main 上包含 V1 packed-key + Triton RN 基础，再增加可选 AIR 路径。
- AIR 前的本分支基线：`49af46f`。关闭 AIR 即可用同一安装做 V1+RN 对照。
- 未包含 values-only sort、selected-row 或 state-copy tiling 三个实验。
- 只替换 `top_p_renorm_prob`；top-k、tree/chain/rejection、state-copy 不变。
  有效 top-k 过滤若被调用，仍使用原来的实现。

算法参考固定版本：
[FlashInfer air_top_p.cuh](https://github.com/flashinfer-ai/flashinfer/blob/c94c1a9e4bf151e56080168ac880ebf0651af317/include/flashinfer/air_top_p.cuh)。
来源说明和 Apache-2.0 许可随 Python 模块分发。

## 算法与取舍

1. 将非负 FP32 概率的位模式映射为降序整数 key。
2. 每轮读取输入，对当前边界前缀中的 token 按 4 bit 建立 16 桶概率质量直方图。
   每个 tile 独立归约并写自己的槽位，没有全局逐 token scatter/atomic。
3. 每行归约 tile 直方图，用 16 元素前缀和定位边界桶，扣除之前桶的质量。
4. 共 8 轮精化得到 FP32 阈值；按原词表顺序统计保留概率，再用 RN 除法输出。

消除了两次全词表排序、全词表 cumsum 和排序索引恢复。当前共 19 次 launch：
8 × (局部直方图 + 边界选择)，随后保留质量、总质量、归一化各一次。
FlashInfer CUDA 使用更宽 radix、共享内存原子操作及边界桶压缩；本版采用 4 bit
稠密局部比较/归约，并**每轮重扫输入，尚未做候选桶压缩**。这是优先验证正确性
与编译可行性的移植版本，可能被重复扫描、局部比较或 launch 开销限制，甚至慢于 V1。
只有设备实测才能决定下一步是压缩边界候选、调整 radix，还是停止此方向。

支持连续 FP32 `[rows, vocab]`、非负有限概率、标量/逐行 top-p。
`rows = batch × draft_tokens`，不限制 batch 为 54，不要求 top-k=1。
block 可为 256/512/1024，要求 `ceil(vocab / block) <= 1024`，超出时公共接口
退回 V1。直接调用实验函数会报错。输入非负且有限是调用方契约，不在热路径
上通过同步设备到主机进行检查。

## 数值语义必须区分

- AIR 目标质量是 `top_p × row_sum`，选择首次达到目标的阈值，并保留该阈值处
  **所有相同概率的 token**。全零行输出零；p=0 保留所有最大值；p=1 保留全部。
- V1 使用 `cumsum - value > top_p` 过滤，比较的是绝对 top-p；等于边界时可能
  额外保留下一 token，重复值也可能被切开。均匀分布的差异可能很大。
- 本移植最终采用 `sum.clamp_min(1e-20)` 和 `tl.div_rn`，保留 RN 归一化方向，
  与 FlashInfer 的 reciprocal multiply/极小值处理也不是逐 bit 相同。
- RN 不保证不同归约顺序得到完全一样的分母或阈值。CPU 大词表检查中出现过
  相对 FP64 参考偏一个边界 token 的情况，脚本始终报告 support 差异。
- 验证要求输出为完整阈值集合、保留所有边界 ties、正确归一化；相对 FP64
  的截断质量误差不得超过行质量的 `2e-6`，TV 不得超过 `2e-5`。
  这些是实验数值门槛，不是整网精度和接受长度等价的保证。

## 获取与安装

先停止服务。在服务器取独立分支，保留当前其他实验目录：

```bash
git clone --branch perf/sampling-air-top-p --single-branch \
  https://github.com/cl-vv-h/sgl-kernel-npu.git sgl-kernel-npu-air
cd sgl-kernel-npu-air
python scripts/install_sampling_air.py
```

HTTPS 不通可把地址换为 `git@github.com:cl-vv-h/sgl-kernel-npu.git`。
安装脚本只覆盖 Python sampling 文件，并打印备份路径、安装目标及 SHA256；
继续使用已安装的可运行 NPU 扩展，不重新编译 C++。如果要完整构建，按仓库
现有 build.sh 文档操作。必须重启服务，已导入模块或已捕获图不会自动更新。

## 先验证，再测性能

先测小 batch，失败会显示最后正在编译/执行的阶段：

```bash
python benchmark/bench_sampling_air.py --batch-sizes 1 --smoke-only
```

然后按真实形状比较完整算子（默认 D=4、p=0.95）：

```bash
python benchmark/bench_sampling_air.py \
  --batch-sizes 1 16 54 --vocab-size 248320 --draft-tokens 4 \
  --blocks 512 --warmup 5 --iterations 30 --graph | tee air_bench.log
```

`--graph` 会执行 graph capture/replay 对照和输出检查；编译器或图接口报错时
不会吞掉错误或伪装为 eager 成功。可先省略它定位普通执行问题。
脚本内部强制 V1 对照使用 TRITON=1、DIV=rn、AIR=0，AIR 直接调用新函数，
还单独检查启用环境变量后的公共接口分发。无需先设置环境变量跑 benchmark。

进一步诊断：

```bash
python benchmark/bench_sampling_air.py --batch-sizes 54 \
  --blocks 256 512 1024 --stage-times --iterations 20
python benchmark/bench_sampling_air.py --batch-sizes 1 \
  --cases softmax peaked uniform ties --iterations 10
python -m pytest -q tests/python/sgl_kernel_npu/test_sampling_air_npu.py
```

阶段计时包含逐阶段同步，首次还含 JIT 编译；**不能相加当成稳态服务耗时**。
`mode=eager/graph` 则在预热后关闭阶段 hook，以交替执行顺序计时并报告完整
算子的同步 wall-time。图重放计时仍含主机调用和同步成本，不是纯设备 kernel time。
合成 softmax 输入仅供筛选；需再用模型真实概率/端到端服务验证。

## 服务开关与回退

新终端中沿用正常 SGLang 启动参数，设置：

```bash
export SGL_KERNEL_NPU_SAMPLING_TRITON=1
export SGL_KERNEL_NPU_SAMPLING_TRITON_DIV=rn
export SGL_KERNEL_NPU_SAMPLING_AIR=1
export SGL_KERNEL_NPU_SAMPLING_AIR_BLOCK=512
```

AIR 当前始终使用 RN，DIV 控制 V1 对照和未替换的 top-k 路径。
不用改 SGLang，不需要新命令行参数；不要叠加另外三个实验分支。
保持请求 temperature=1、top_p=0.95 及原 MTP 配置，并核实请求的 DP/device。
默认 AIR=0；回退 V1+RN：

```bash
export SGL_KERNEL_NPU_SAMPLING_AIR=0
# 重新启动服务，以重建执行图。
```

固定 workload、实际 batch、输出长度与机器负载，再比较 TPOT、吞吐、平均接受
长度和输出质量。若只有微基准变快而 TPOT 不变，不宣称整网优化成功。

## 当前验证边界

开发机只能执行 CPU 算法模拟与接口分发检查。实际 Triton kernel 通过 AST
适配器执行，与独立 FP64 排序参考比较；这不验证 Ascend 编译器、UB/寄存器
资源分配、设备上的归约顺序或图兼容性。NPU 编译、性能和整网收益均待服务器验证。
