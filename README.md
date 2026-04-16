# llama.cpp ROCm/gfx906 Support + Multi-GPU Tensor Parallel

本仓库包含三个主要组件：

1. **llama.cpp ROCm/gfx906 编译支持** - 已编译的二进制文件和补丁
2. **ggml-rccl 多 GPU 通信模块** - 张量并行通信后端
3. **ggml-tensor-parallel 张量并行模块** - 双 GPU 协作推理核心
4. **multi-gpu-launcher 多 GPU 启动器** - 数据并行方案

---

## 编译说明 (ROCm 6.3 + gfx906)

### 问题

ROCm 6.3 的 clang 18 编译器默认不支持 gfx906 架构，编译时会报错：
```
clang: error: unsupported CUDA gpu architecture: gfx906
```

### 解决方案

通过添加 C++ 标准库 include 路径，可以成功编译：

```bash
# 设置环境
export ROCM_PATH=/opt/rocm-6.3.0
export HIP_PATH=$ROCM_PATH
export CMAKE_PREFIX_PATH="$ROCM_PATH/lib/cmake"
export CPLUS_INCLUDE_PATH="/usr/include/c++/11:/usr/include/x86_64-linux-gnu/c++/11"
export HIPCXX="$ROCM_PATH/lib/llvm/bin/clang++"

# 配置编译
mkdir -p build && cd build
cmake .. \
  -DGGML_HIP=ON \
  -DGPU_TARGETS=gfx906 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_FLAGS="-fPIC -I/usr/include/c++/11 -I/usr/include/x86_64-linux-gnu/c++/11"

# 编译
make -j$(nproc)
```

或使用提供的构建脚本：

```bash
./build-rocm63-gfx906.sh
```

### 验证编译成功

```bash
./build/bin/llama-cli --help
# 应该显示：found 2 ROCm devices
```

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

## ggml-tensor-parallel: 张量并行核心

### 概述

张量并行实现，参考 vLLM-gfx906 架构，在 attention 和 FFN 层之间插入 AllReduce 同步。

### 文件结构

```
ggml-tensor-parallel/
├── ggml-tensor-parallel.h     # 主头文件
├── ggml-tensor-parallel.cpp    # 核心实现
├── ggml-tp-comm.h              # 通信原语
├── ggml-tp-comm.cpp           # Ring AllReduce 实现
├── ggml-tp-shard.h            # 权重分片
├── ggml-tp-shard.cpp          # 分片实现
└── CMakeLists.txt             # 构建配置
```

### 架构

```
GPU 0: [Q,K,V W0/2] ──┐
                       ├── AllReduce ──> attention output
GPU 1: [Q,K,V W1/2] ──┘

attention output ──> [Gate/Up W0/2] ──┐
                                     ├── AllReduce ──> FFN output
                  [Gate/Up W1/2] ─────┘
```

### API

```cpp
#include "ggml-tensor-parallel.h"

// 初始化
int devices[] = {0, 1};
ggml_tp_init(&tp, 2, devices);

// 张量并行矩阵乘法 (自动 all_reduce)
struct ggml_tensor * output = ggml_cuda_tp_mul_mat(&tp, ctx, weight, input, true);

// Barrier 同步
ggml_tp_barrier(&tp);
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
输入: "The capital of France is"
输出: "The capital of France is Paris." ✅
速度: 56.6 tokens/s
分割模式: --split-mode layer (层分割)
```

**GPU 使用:**
- GPU 0: 31% VRAM
- GPU 1: 31% VRAM

**结论:** 双 GPU 层分割 (Pipeline Parallelism) 工作正常！

---

## 使用示例

```bash
# 设置环境
export ROCM_PATH=/opt/rocm-6.3.0
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH

# 双 GPU 层分割推理
./llama-cli -m model.gguf -p "Hello" \
  --split-mode layer \
  -c 512 -t 16

# 指定 GPU 分片比例
./llama-cli -m model.gguf -p "Hello" \
  --split-mode layer \
  --tensor-split 60,40 \
  -c 512 -t 16
```

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
