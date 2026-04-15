# llama.cpp ROCm/gfx906 Support + Multi-GPU Communication

本仓库包含两个主要组件：

1. **llama.cpp ROCm/gfx906 编译支持** - 已编译的二进制文件和补丁
2. **ggml-rccl 多 GPU 通信模块** - 张量并行通信后端

---

## ggml-rccl: 多 GPU 通信模块

### 概述

这个模块为 llama.cpp 提供多 GPU 通信支持，支持两种实现：

1. **RCCL (ROCm Collective Communications Library)** - 当系统支持时使用
2. **HIP P2P/Ring AllReduce** - 作为 RCCL 的回退方案

## 文件结构

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

## API

### 初始化

```cpp
#include "ggml-rccl.h"

// 初始化双 GPU 通信
int devices[] = {0, 1};
GGMLRCCLComm::instance().init(2, devices);

// 销毁
GGMLRCCLComm::instance().finalize();
```

### AllReduce

```cpp
// 对 device 上的数据执行 AllReduce (求和)
GGMLRCCLComm::instance().all_reduce(
    sendBuf,    // 发送缓冲区
    recvBuf,    // 接收缓冲区
    count,      // 元素数量
    GGML_TYPE_F16,  // 数据类型
    GGML_OP_SUM,    // 操作 (求和)
    stream      // HIP stream (可为 nullptr)
);
```

### AllGather

```cpp
// 收集所有 GPU 上的数据
GGMLRCCLComm::instance().all_gather(
    sendBuf,
    recvBuf,    // 大小需要是 count * nDevices
    count,
    GGML_TYPE_F16,
    stream
);
```

### Broadcast

```cpp
// 从 root GPU 广播数据到所有 GPU
GGMLRCCLComm::instance().broadcast(
    sendBuf,
    recvBuf,
    count,
    GGML_TYPE_F16,
    rootDevice,  // 源设备
    stream
);
```

## 构建

### 依赖

- ROCm 6.1+
- HIP (HIPCC 或 Clang with HIP 支持)
- 可选: RCCL (提供 NCCL API 支持)

### CMake 配置

```bash
cd llama.cpp
mkdir build && cd build
cmake .. \
    -DGGML_HIP=ON \
    -DCMAKE_HIP_COMPILER=/path/to/clang++ \
    -DROCM_PATH=/opt/rocm-6.3.0
make ggml-rccl
```

## 测试

### Ring AllReduce 测试

```bash
cd ggml-rccl/test
clang++-wrapper -x hip -c test_ring.cpp -o test_ring.o
clang++-wrapper test_ring.o -L/opt/rocm-6.3.0/lib -lamdhip64 -o test_ring
LD_LIBRARY_PATH=/opt/rocm-6.3.0/lib ./test_ring 1048576
```

### 预期输出

```
=== HIP Ring AllReduce Test ===
GPU 0: AMD Radeon (TM) Pro VII (16.0 GB)
  -> Can P2P: NO (using host memory fallback)
GPU 1: AMD Radeon (TM) Pro VII (16.0 GB)
  -> Can P2P: NO (using host memory fallback)

[Worker 0] Result correct! First 10: 3.0 3.0 3.0 3.0 3.0 3.0 3.0 3.0 3.0 3.0
[Worker 1] Result correct! First 10: 3.0 3.0 3.0 3.0 3.0 3.0 3.0 3.0 3.0 3.0
```

## 限制

### gfx906 (MI50) 限制

- MI50 不支持 GPUDirect P2P
- 通信使用 Host 内存回退，延迟较高 (~770ms for 1M elements)
- 建议使用支持 XGMI 的 AMD GPU (如 MI100, MI200, MI300)

### RCCL 兼容性

RCCL 在某些内核配置下可能无法工作：
- 需要 `iommu=pt` 内核参数
- 如果 RCCL 初始化失败，模块会自动回退到 HIP P2P/Ring 实现

## 与 llama.cpp 集成

下一步计划：
1. 集成到 `ggml-hip` 后端
2. 在 attention 层后添加 AllReduce
3. 实现张量分片

## 许可

遵循 llama.cpp 项目许可 (MIT)
