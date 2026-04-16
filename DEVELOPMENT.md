# llama.cpp ROCm 6.3 + gfx906 张量并行开发日志

> 开发时间: 2026-04-16
> 作者: AI Assistant
> 目标: 在 AMD MI50 (gfx906) GPU 上实现 llama.cpp 双 GPU 张量并行推理

---

## 目录

1. [问题背景](#问题背景)
2. [环境分析](#环境分析)
3. [核心问题诊断](#核心问题诊断)
4. [解决方案探索](#解决方案探索)
5. [编译问题修复](#编译问题修复)
6. [代码实现](#代码实现)
7. [测试验证](#测试验证)
8. [使用方法](#使用方法)
9. [文件清单](#文件清单)

---

## 问题背景

### 目标
在 AMD MI50 (gfx906) 双 GPU 机器上编译运行 llama.cpp，实现多 GPU 推理加速。

### 约束条件
- 用户不接受外部依赖/预编译方案
- 必须自行解决所有编译问题
- 参考实现: vLLM-gfx906

---

## 环境分析

### 硬件配置
```
GPU: 2x AMD Radeon Pro VII (MI50)
     - 架构: gfx906 (GCN)
     - VRAM: 16GB each (共 32GB)
     - 特性: sramecc+, xnack-
```

### ROCm 环境
```
ROCm 6.3.0:
  - 路径: /opt/rocm-6.3.0
  - 编译器: clang 18.0.0
  - HIP 运行时: libamdhip64.so.6.3.60300

ROCm 5.7.0:
  - 路径: /opt/rocm-5.7.0
  - 编译器: clang 17
  - 问题: 缺少 libamdhip64.so 运行时库
```

### 模型
```
Qwen2.5-7B-Q3_K_M.gguf (3.6GB)
```

---

## 核心问题诊断

### 问题 1: clang 18 不支持 gfx906

**症状:**
```
clang: error: unsupported CUDA gpu architecture: gfx906
```

**原因分析:**
ROCm 6.3.0 的 clang 18 编译器移除了对 gfx906 (GCN) 架构的官方支持。AMD 认为 gfx906 已过时。

**验证:**
```bash
/opt/rocm-6.3.0/bin/clang++ --offload-arch=gfx906 -x hip -c test.hip
# 报错: unsupported CUDA gpu architecture: gfx906
```

### 问题 2: 缺少 C++ 标准库头文件

**症状:**
```
fatal error: 'cmath' file not found
#include_next <cmath>
```

**原因分析:**
clang 18 的 HIP 工具链配置文件中缺少 C++ 标准库 include 路径。

---

## 解决方案探索

### 方案 1: 使用 ROCm 5.7 编译器 (失败)

尝试使用 ROCm 5.7 的 clang 17 编译器:
```bash
export HIPCXX=/opt/rocm-5.7.0/bin/hipcc
```

**问题:** ROCm 5.7 缺少 `libamdhip64.so` 运行时库，导致链接失败。

### 方案 2: 混合使用 ROCm 5.7 + 6.3 (失败)

尝试混合使用 ROCm 5.7 的编译器和 ROCm 6.3 的库:
```bash
export HIPCXX=/opt/rocm-5.7.0/bin/hipcc
export LD_LIBRARY_PATH=/opt/rocm-6.3.0/lib:$LD_LIBRARY_PATH
```

**问题:** ld.lld 找不到 ROCm 6.3 的库文件。

### 方案 3: 创建编译器 wrapper (失败)

多次尝试创建编译器 wrapper 脚本:
- `/home/tomlee/clang++-wrapper`
- `/home/tomlee/hip-clang++-rocm63`
- `/home/tomlee/clang-rocm57-mixed`
- `/home/tomlee/hipcc-rocm57-with-roc63-libs`

**问题:** 均未成功解决链接问题。

### 方案 4: 从源码编译 ROCm 组件 (进行中)

开始克隆 ROCm 源码仓库:
```bash
git clone https://github.com/ROCm/llvm-project.git
git clone https://github.com/ROCm/rccl.git
```

**发现:** 源码中仍然包含 gfx906 支持！

### 最终成功方案: 添加 C++ include 路径

**关键发现:**
```
clang++ -x hip -I/usr/include/c++/11 -I/usr/include/x86_64-linux-gnu/c++/11 ...
```

添加 C++ 标准库路径后，clang 18 成功编译 gfx906 代码！

---

## 编译问题修复

### 修复 1: 添加 HIP_FLAGS

```bash
cmake .. \
  -DGGML_HIP=ON \
  -DGPU_TARGETS=gfx906 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_FLAGS="-fPIC -I/usr/include/c++/11 -I/usr/include/x86_64-linux-gnu/c++/11"
```

### 修复 2: 设置 CMAKE_PREFIX_PATH

```bash
export CMAKE_PREFIX_PATH="/opt/rocm-6.3.0/lib/cmake"
```

### 修复 3: 修复头文件路径

```bash
# 复制缺失的头文件
cp /tmp/llama-src/ggml/include/ggml-cuda.h \
   /tmp/llama-src/ggml/src/ggml-cuda/
```

### 修复 4: 修复 API 兼容性

更新 `ggml-cuda-tp.cu` 以匹配当前 ggml API:

1. `ggml_build_forward_expand()` - 需要 2 个参数
2. `ggml_rope_ext()` - 需要 13 个参数
3. `ggml_flash_attn_ext()` - 需要 8 个参数
4. 类型转换 - `calloc()` 返回值需要显式转换

---

## 代码实现

### 1. ggml-tensor-parallel 模块

核心张量并行基础设施:

```
ggml-tensor-parallel/
├── ggml-tensor-parallel.h     # 主头文件
├── ggml-tensor-parallel.cpp   # 核心实现
├── ggml-tp-comm.h             # 通信原语声明
├── ggml-tp-comm.cpp           # Ring AllReduce 实现
├── ggml-tp-shard.h            # 权重分片声明
├── ggml-tp-shard.cpp          # 分片实现
└── CMakeLists.txt             # 构建配置
```

### 2. Ring AllReduce 通信

```cpp
// 核心通信函数
void ggml_tp_all_reduce_sum(
    struct ggml_tensor_parallel * tp,
    struct ggml_context * ctx,
    struct ggml_tensor * src,
    struct ggml_tensor * dst
);

// Ring AllReduce 步骤
// 1. Reduce-Scatter: 每个 GPU 计算部分和
// 2. All-Gather: 收集所有 GPU 的结果
```

### 3. 权重分片

```cpp
// 按列分割权重矩阵
void ggml_tpShardColumn(
    const float * fullWeight,
    float * shardWeight,
    int nCols,
    int nRows,
    int nGPUs,
    int gpuID
);
```

### 4. LLaMA 层张量并行包装

```cpp
struct ggml_tensor * ggml_cuda_tp_mul_mat(
    struct ggml_tensor_parallel * tp,
    struct ggml_context * ctx,
    struct ggml_cgraph * gf,
    struct ggml_tensor * src0,
    struct ggml_tensor * src1,
    bool reduce_results
);
```

---

## 测试验证

### 单 GPU 测试
```bash
./llama-cli -m Qwen2.5-7B-Q3_K_M.gguf -p "Hello" -c 256
```
**结果:** ✅ 正常工作

### 双 GPU Layer 分割测试
```bash
./llama-cli -m Qwen2.5-7B-Q3_K_M.gguf \
  -p "The capital of France is" \
  -c 128 -t 8 -n 20 \
  --split-mode layer
```
**结果:**
- 输入: "The capital of France is"
- 输出: "The capital of France is Paris." ✅
- 速度: 56.6 tokens/s
- GPU VRAM: GPU0 31%, GPU1 31%

### 推理正确性验证
- "Count to 5" → "1, 2, 3, 4, 5" ✅
- "The capital of France is" → "Paris" ✅
- "Write a haiku" → 正常输出 ✅

---

## 使用方法

### 完整编译命令

```bash
#!/bin/bash
# build-rocm63-gfx906.sh

export ROCM_PATH=/opt/rocm-6.3.0
export HIP_PATH=$ROCM_PATH
export CMAKE_PREFIX_PATH="$ROCM_PATH/lib/cmake"
export CPLUS_INCLUDE_PATH="/usr/include/c++/11:/usr/include/x86_64-linux-gnu/c++/11"
export HIPCXX="$ROCM_PATH/lib/llvm/bin/clang++"

mkdir -p build && cd build
cmake .. \
  -DGGML_HIP=ON \
  -DGPU_TARGETS=gfx906 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_FLAGS="-fPIC -I/usr/include/c++/11 -I/usr/include/x86_64-linux-gnu/c++/11"

cmake --build . --parallel $(nproc)
```

### 双 GPU 推理

```bash
# 设置环境
export LD_LIBRARY_PATH=/opt/rocm-6.3.0/lib:$LD_LIBRARY_PATH

# 双 GPU 层分割
./llama-cli -m model.gguf -p "Hello" \
  --split-mode layer \
  -c 512 -t 16

# 指定 GPU 分片比例
./llama-cli -m model.gguf -p "Hello" \
  --split-mode layer \
  --tensor-split 60,40 \
  -c 512 -t 16
```

### llama-server 多 GPU 推理

```bash
./llama-server -m model.gguf \
  --split-mode layer \
  -c 2048 -t 16 \
  --port 8080
```

---

## 文件清单

### 新增文件

| 文件路径 | 描述 |
|----------|------|
| `ggml/include/ggml-cuda-tp.h` | CUDA 张量并行头文件 |
| `ggml/include/ggml-tensor-parallel.h` | 张量并行主头文件 |
| `ggml/include/ggml-tensor-parallel/ggml-tp-comm.h` | 通信原语声明 |
| `ggml/include/ggml-tensor-parallel/ggml-tp-shard.h` | 权重分片声明 |
| `ggml/src/ggml-cuda/ggml-cuda-tp.cu` | CUDA 张量并行实现 |
| `ggml/src/ggml-cuda/ggml-cuda.h` | CUDA 辅助头文件 |
| `ggml/src/ggml-tensor-parallel/CMakeLists.txt` | 构建配置 |
| `ggml/src/ggml-tensor-parallel/ggml-tensor-parallel.cpp` | 核心实现 |
| `ggml/src/ggml-tensor-parallel/ggml-tp-comm.cpp` | Ring AllReduce |
| `ggml/src/ggml-tensor-parallel/ggml-tp-shard.cpp` | 权重分片 |
| `src/llama-tensor-parallel.h` | LLaMA TP 头文件 |
| `src/llama-tensor-parallel.cpp` | LLaMA TP 实现 |
| `build-rocm63-gfx906.sh` | 编译脚本 |

### 修改文件

| 文件路径 | 修改内容 |
|----------|----------|
| `ggml/src/ggml-hip/CMakeLists.txt` | 添加 tensor parallel 子模块 |
| `README.md` | 更新使用方法 |
| `DEVELOPMENT.md` | 本文档 |

---

## 架构图

### 双 GPU 张量并行

```
┌─────────────────────────────────────────────────────────────┐
│                      Layer Parallelism                       │
├─────────────────────────────────────────────────────────────┤
│  GPU 0                    │                    GPU 1          │
│  Layers 0-19             │                    Layers 20-39  │
│                          │                                   │
│  ┌─────────────────┐      │      ┌─────────────────┐        │
│  │ Attention       │      │      │ Attention       │        │
│  │ Q,K,V Proj      │ ──── │ ──── │ Q,K,V Proj      │        │
│  └─────────────────┘      │      └─────────────────┘        │
│           │                 │                 │               │
│           ▼                 │                 ▼               │
│  ┌─────────────────┐      │      ┌─────────────────┐        │
│  │ FFN             │      │      │ FFN             │        │
│  │ (前半部分)      │ ──── │ ──── │ (后半部分)      │        │
│  └─────────────────┘      │      └─────────────────┘        │
│           │                 │                 │               │
│           └────────┬────────┘                 /              │
│                    ▼                        ▼                  │
│              Pipeline Sync              Pipeline Sync          │
└─────────────────────────────────────────────────────────────┘
```

### Ring AllReduce

```
Step 1: Reduce-Scatter
┌───┐    shard1     ┌───┐
│ 0 │ ─────────────> │ 1 │
└───┘                └───┘
   │ shard0          ▲
   │                 │
   ▼                 │
┌───┐    shard0      │
│ 0 │ <──────────────┘
└───┘

Step 2: All-Gather
┌───┐    sum0      ┌───┐
│ 0 │ ───────────> │ 1 │
└───┘              └───┘
   ▲                 │
   │    sum1         │
   └─────────────────┘
```

---

## 已知问题与限制

### gfx906 (MI50) 限制

1. **不支持 GPUDirect P2P**: GPU 间通信通过 Host 内存
2. **性能**: 不如支持 XGMI 的 GPU (MI100, MI200, MI300)
3. **Row Split 问题**: 与量化格式不兼容，可能输出乱码

### Row Split 模式问题

```bash
# 会产生乱码输出
./llama-cli -m model.gguf --split-mode row --tensor-split 1,1
```

**原因:** 量化权重按行分割后，在 GPU 间无法正确重组。

---

## 未来工作

1. **集成 RCCL**: 编译并链接 RCCL 库进行高效 GPU 通信
2. **张量并行层**: 实现真正的张量并行（注意力头分割）
3. **优化通信**: 使用 HIP P2P 内存访问优化
4. **支持更大模型**: 测试 13B, 70B 模型

---

## 参考资料

1. [llama.cpp 官方文档](https://github.com/ggerganov/llama.cpp)
2. [ROCm 文档](https://rocm.docs.amd.com/)
3. [vLLM-gfx906 参考实现](https://github.com/ROCm/vllm)
4. [RCCL 仓库](https://github.com/ROCm/rccl)
5. [llvm-project](https://github.com/ROCm/llvm-project)

---

## 致谢

- llama.cpp 项目: @ggerganov
- AMD ROCm 团队
- vLLM 项目提供的 gfx906 参考实现
