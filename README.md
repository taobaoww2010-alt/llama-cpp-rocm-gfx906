# llama.cpp ROCm/gfx906 Support + Multi-GPU Communication

本仓库包含两个主要组件：

1. **llama.cpp ROCm/gfx906 编译支持** - 已编译的二进制文件和补丁
2. **ggml-rccl 多 GPU 通信模块** - 张量并行通信后端
3. **multi-gpu-launcher 多 GPU 启动器** - 数据并行方案

---

## 快速开始

### 双 GPU 并行推理

```bash
# 启动 GPU 0 服务器
screen -S gpu0
export LD_LIBRARY_PATH=/path/to/libs:$LD_LIBRARY_PATH
export ROCM_PATH=/opt/rocm-6.3.0
export HIP_VISIBLE_DEVICES=0
./llama-server -m model.gguf -ngl 99 -np 2 --port 8080

# 新建窗口，启动 GPU 1 服务器
screen -S gpu1
export LD_LIBRARY_PATH=/path/to/libs:$LD_LIBRARY_PATH
export ROCROCM_PATH=/opt/rocm-6.3.0
export HIP_VISIBLE_DEVICES=1
./llama-server -m model.gguf -ngl 99 -np 2 --port 8081
```

### 验证双 GPU 工作

```bash
# GPU 0 处理请求
curl http://localhost:8080/v1/chat/completions \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":20}'

# GPU 1 处理请求
curl http://localhost:8081/v1/chat/completions \
  -d '{"messages":[{"role":"user","content":"Hi"}],"max_tokens":20}'

# 检查 GPU 使用
rocm-smi --showmemuse
```

---

## ggml-rccl: 多 GPU 通信模块

### 概述

这个模块为 llama.cpp 提供多 GPU 通信支持，支持两种实现：

1. **RCCL (ROCm Collective Communications Library)** - 当系统支持时使用
2. **HIP P2P/Ring AllReduce** - 作为 RCCL 的回退方案

### 文件结构

```
ggml-rccl/
├── ggml-rccl.h       # 主头文件，API 声明
├── ggml-rccl.cpp     # 核心实现
├── hip-comm.h        # HIP 通信辅助 (可选)
├── hip-comm.cpp      # HIP 通信实现 (可选)
├── CMakeLists.txt    # CMake 构建配置
└── test/
    ├── test_ring.cpp # Ring AllReduce 测试
    └── test_rccl.cpp # RCCL 测试 (需要 RCCL)
```

### API

```cpp
#include "ggml-rccl.h"

// 初始化双 GPU 通信
int devices[] = {0, 1};
GGMLRCCLComm::instance().init(2, devices);

// AllReduce
GGMLRCCLComm::instance().all_reduce(sendBuf, recvBuf, count, GGML_TYPE_F16, GGML_OP_SUM, stream);
```

---

## 测试结果 (2026-04-16)

### 双 GPU 并行推理测试

**硬件配置:**
- GPU: 2x AMD Radeon Pro VII (gfx906, 16GB VRAM)
- ROCm: 6.3.0
- 模型: Qwen2.5-7B-Q3_K_M.gguf (3.6GB)

**测试结果:**
```
GPU 0: 处理请求 "Count to 5"  -> 0.49 tokens/s
GPU 1: 处理请求 "Count to 10" -> 0.49 tokens/s

GPU 内存使用:
- GPU 0: 29% -> 31% (处理中)
- GPU 1: 29% -> 31% (处理中)
```

**结论:** 双 GPU 数据并行方案工作正常！

---

## multi-gpu-launcher

提供脚本简化多 GPU 启动流程：

```bash
./multi-gpu-launcher.sh -m model.gguf -n 4 -g 0,1
```

---

## 限制

### gfx906 (MI50) 限制

- MI50 不支持 GPUDirect P2P
- 通信使用 Host 内存回退
- 建议使用支持 XGMI 的 AMD GPU (如 MI100, MI200, MI300)

### 当前方案限制

- 数据并行：每个 GPU 运行完整模型
- 无张量并行：GPU 间无通信
- 适合多并发请求场景

---

## 许可

遵循 llama.cpp 项目许可 (MIT)
