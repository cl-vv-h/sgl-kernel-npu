# Python tests

公共算子框架 UT 见 [framework/README.md](framework/README.md)。在 NPU 环境执行：

```bash
python -m pytest tests/python/framework -v -rs -x --npu-device=0
```

本批 UT 基于 `Todobe:pr/op-refactor` 的 `5e36a425888b3eee13cb149a3ebe2e22dbedca1e`。
其余目录保留该基线的测试；`--npu-device` 当前仅供 framework UT 使用。
