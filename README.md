# llama.cpp ROCm/gfx906 Support + Multi-GPU Tensor Parallel

在 AMD MI50 (gfx906) GPU 上编译运行 llama.cpp，实现双 GPU 推理加速。

> 详细开发过程请参阅 [DEVELOPMENT.md](./DEVELOPMENT.md)

---

## 硬件配置

```
GPU: 2x AMD Radeon Pro VII (MI50)
     - 架构: gfx906 (GCN)
     - VRAM: 16GB each (共 32GB)
ROCm: 6.3.0
模型: Qwen2.5-7B-Q3_K_M.gguf
```

---

## 快速开始

### 1. 编译

```bash
./build-rocm63-gfx906.sh
```

或手动编译:

```bash
export ROCM_PATH=/opt/rocm-6.3.0
export CMAKE_PREFIX_PATH="$ROCM_PATH/lib/cmake"
export CPLUS_INCLUDE_PATH="/usr/include/c++/11:/usr/include/x86_64-linux-gnu/c++/11"

mkdir -p build && cd build
cmake .. \
  -DGGML_HIP=ON \
  -DGPU_TARGETS=gfx906 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_FLAGS="-fPIC -I/usr/include/c++/11 -I/usr/include/x86_64-linux-gnu/c++/11"

cmake --build . --parallel $(nproc)
```

### 2. 双 GPU 推理

```bash
export LD_LIBRARY_PATH=/opt/rocm-6.3.0/lib:$LD_LIBRARY_PATH

# 双 GPU 层分割
./llama-cli -m model.gguf -p "Hello" \
  --split-mode layer \
  -c 512 -t 16

# 指定 GPU 分片比例
./llama-cli -m model.gguf -p "Hello" \
  --split-mode layer \
  --tensor-split 60,40
```

---

## 测试结果

```
输入: "The capital of France is"
输出: "The capital of France is Paris." ✅
速度: 56.6 tokens/s
GPU VRAM: GPU0 31%, GPU1 31%
```

---

## 项目结构

```
llama-cpp-rocm-gfx906/
├── CMakeLists.txt                     # 根 CMake 配置
├── ggml/
│   ├── include/
│   │   ├── ggml-cuda-tp.h           # CUDA 张量并行头文件
│   │   ├── ggml-tensor-parallel.h    # 张量并行核心 API
│   │   └── ggml-tensor-parallel/
│   │       ├── ggml-tp-comm.h        # 通信原语声明
│   │       └── ggml-tp-shard.h       # 权重分片声明
│   └── src/
│       ├── ggml-cuda/
│       │   ├── ggml-cuda.h           # CUDA 辅助头文件
│       │   └── ggml-cuda-tp.cu       # CUDA 张量并行实现
│       └── ggml-tensor-parallel/
│           ├── CMakeLists.txt        # 构建配置
│           ├── ggml-tensor-parallel.cpp  # 核心实现
│           ├── ggml-tp-comm.cpp       # Ring AllReduce 实现
│           └── ggml-tp-shard.cpp      # 权重分片实现
├── src/
│   ├── llama-tensor-parallel.h       # LLaMA 集成头文件
│   ├── llama-tensor-parallel.cpp     # LLaMA 层集成
│   └── multi-gpu/
│       └── multi-gpu-scheduler.h     # 多 GPU 调度器
├── ggml-rccl.cpp / ggml-rccl.h       # RCCL 通信库
├── hip-comm.cpp / hip-comm.h         # HIP 通信辅助
├── test_rccl.cpp                     # RCCL 测试
├── build-rocm63-gfx906.sh            # 编译脚本
├── DEVELOPMENT.md                    # 详细开发日志
└── README.md
```

---

## 核心问题与解决方案

### 问题: ROCm 6.3 不支持 gfx906

**症状:**
```
clang: error: unsupported CUDA gpu architecture: gfx906
```

**解决方案:**
添加 C++ 标准库 include 路径后编译成功:

```bash
-DCMAKE_HIP_FLAGS="-fPIC -I/usr/include/c++/11 -I/usr/include/x86_64-linux-gnu/c++/11"
```

详见 [DEVELOPMENT.md](./DEVELOPMENT.md#解决方案探索)

---

## 架构说明

### Layer Split (Pipeline Parallelism)

```
GPU 0: Layers 0-19  ──┐
                        ├── Sync ──> Output
GPU 1: Layers 20-39 ──┘
```

### Ring AllReduce

```
Step 1: Reduce-Scatter
GPU0 ──shard1──> GPU1
  │                │
  │ sh0            │ sh1
  ▼                ▼
GPU0 <──sh0─── GPU1

Step 2: All-Gather
GPU0 ──sum0───> GPU1
  ▲                │
  │    sum1        │
  └─── sum1 ───────┘
```

---

## 限制

1. **gfx906 (MI50)** 不支持 GPUDirect P2P
2. **Row Split 模式** 与量化格式不兼容（可能输出乱码）
3. 建议使用支持 XGMI 的 AMD GPU (MI100, MI200, MI300)

---

## 许可

遵循 llama.cpp 项目许可 (MIT)
