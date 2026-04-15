#include "ggml-rccl.h"

#include "../ggml-impl.h"

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <string.h>

#define HIP_CHECK(call)                                                        \
    do {                                                                       \
        hipError_t err = call;                                                  \
        if (err != hipSuccess) {                                               \
            fprintf(stderr, "HIP error at %s:%d: %s\n",                         \
                    __FILE__, __LINE__, hipGetErrorString(err));                \
        }                                                                      \
    } while (0)

#if GGML_RCCL_NCCL_AVAILABLE
#define RCCL_CHECK(call)                                                       \
    do {                                                                       \
        ncclResult_t err = call;                                                \
        if (err != ncclSuccess) {                                              \
            fprintf(stderr, "RCCL error at %s:%d: %s\n",                        \
                    __FILE__, __LINE__, ncclGetErrorString(err));               \
        }                                                                      \
    } while (0)
#endif

static size_t ggml_type_sizeof(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:  return 4;
        case GGML_TYPE_F16:  return 2;
        case GGML_TYPE_BF16: return 2;
        case GGML_TYPE_I8:   return 1;
        case GGML_TYPE_I32:  return 4;
        case GGML_TYPE_I64:  return 8;
        default:             return 2;
    }
}

static const char* ggml_type_name(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:  return "f32";
        case GGML_TYPE_F16:  return "f16";
        case GGML_TYPE_BF16: return "bf16";
        case GGML_TYPE_I8:   return "i8";
        case GGML_TYPE_I32:  return "i32";
        case GGML_TYPE_I64:  return "i64";
        default:             return "unknown";
    }
}

GGMLRCCLComm::GGMLRCCLComm() :
    initialized_(false),
    use_rccl_(false),
    nDevices_(0),
    rank_(-1) {
}

GGMLRCCLComm::~GGMLRCCLComm() {
    if (initialized_) {
        finalize();
    }
}

GGMLRCCLComm& GGMLRCCLComm::instance() {
    static GGMLRCCLComm instance_;
    return instance_;
}

void GGMLRCCLComm::init(int nDevices, const int* deviceIds) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (initialized_) {
        fprintf(stderr, "GGMLRCCLComm already initialized\n");
        return;
    }

    if (nDevices < 1) {
        fprintf(stderr, "GGMLRCCLComm: invalid nDevices=%d\n", nDevices);
        return;
    }

    nDevices_ = nDevices;
    devices_.resize(nDevices);
    peer_access_.resize(nDevices * nDevices, false);
    host_buffers_.resize(nDevices);
    step_mutex_.resize(nDevices);
    step_cv_.resize(nDevices);
    step_counts_.resize(nDevices, 0);
    step_generations_.resize(nDevices, 0);

#if GGML_RCCL_NCCL_AVAILABLE
    nccl_comms_.resize(nDevices);
#endif

    for (int i = 0; i < nDevices; i++) {
        devices_[i] = deviceIds[i];

        for (int j = 0; j < nDevices; j++) {
            if (i != j) {
                int canAccess = 0;
                HIP_CHECK(hipDeviceCanAccessPeer(&canAccess, deviceIds[i], deviceIds[j]));

                if (canAccess) {
                    HIP_CHECK(hipSetDevice(deviceIds[i]));
                    hipError_t err = hipDeviceEnablePeerAccess(deviceIds[j], 0);
                    if (err == hipSuccess || err == hipErrorPeerAccessAlreadyEnabled) {
                        peer_access_[i * nDevices_ + j] = true;
                        fprintf(stderr, "GGMLRCCLComm: P2P enabled GPU %d -> GPU %d\n", i, j);
                    }
                }
            }
        }
    }

#if GGML_RCCL_NCCL_AVAILABLE
    if (use_rccl_) {
        RCCL_CHECK(ncclGetUniqueId(&uniqueId_));

        HIP_CHECK(hipSetDevice(deviceIds[0]));
        RCCL_CHECK(ncclCommInitRank(&nccl_comms_[0], nDevices, uniqueId_, 0));
        rank_ = 0;

        for (int i = 1; i < nDevices; i++) {
            HIP_CHECK(hipSetDevice(deviceIds[i]));
            RCCL_CHECK(ncclCommInitRank(&nccl_comms_[i], nDevices, uniqueId_, i));
        }
    }
#endif

    for (int i = 0; i < nDevices; i++) {
        HIP_CHECK(hipHostMalloc(&host_buffers_[i], 64 * 1024 * 1024, hipHostMallocDefault));
    }

    initialized_ = true;
    fprintf(stderr, "GGMLRCCLComm: initialized %d devices, mode=%s\n",
            nDevices, use_rccl_ ? "RCCL" : "HIP-P2P/Ring");
}

void GGMLRCCLComm::finalize() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!initialized_) {
        return;
    }

#if GGML_RCCL_NCCL_AVAILABLE
    if (use_rccl_) {
        for (int i = 0; i < nDevices_; i++) {
            if (nccl_comms_[i]) {
                ncclCommDestroy(nccl_comms_[i]);
                nccl_comms_[i] = nullptr;
            }
        }
    }
#endif

    for (int i = 0; i < nDevices_; i++) {
        for (int j = 0; j < nDevices_; j++) {
            if (peer_access_[i * nDevices_ + j]) {
                HIP_CHECK(hipSetDevice(devices_[i]));
                HIP_CHECK(hipDeviceDisablePeerAccess(devices_[j]));
            }
        }
        if (host_buffers_[i]) {
            HIP_CHECK(hipHostFree(host_buffers_[i]));
            host_buffers_[i] = nullptr;
        }
    }

    devices_.clear();
    peer_access_.clear();
    host_buffers_.clear();
    step_mutex_.clear();
    step_cv_.clear();
    step_counts_.clear();
    step_generations_.clear();

#if GGML_RCCL_NCCL_AVAILABLE
    nccl_comms_.clear();
#endif

    initialized_ = false;
    nDevices_ = 0;
    rank_ = -1;

    fprintf(stderr, "GGMLRCCLComm: finalized\n");
}

void GGMLRCCLComm::barrier_wait(int step) {
    int tid;
    HIP_CHECK(hipGetDevice(&tid));

    std::unique_lock<std::mutex> lock(step_mutex_[tid]);
    int gen = step_generations_[tid];

    if (++step_counts_[tid] == nDevices_) {
        step_generations_[tid]++;
        step_counts_[tid] = 0;
        for (int i = 0; i < nDevices_; i++) {
            step_cv_[i].notify_all();
        }
    } else {
        step_cv_[tid].wait(lock, [this, tid, gen] {
            return gen != step_generations_[tid];
        });
    }
}

void GGMLRCCLComm::sync_device(int deviceId) {
    HIP_CHECK(hipSetDevice(deviceId));
    HIP_CHECK(hipDeviceSynchronize());
}

void GGMLRCCLComm::sync_all() {
    for (int i = 0; i < nDevices_; i++) {
        sync_device(i);
    }
}

void GGMLRCCLComm::all_reduce(const void* sendBuf, void* recvBuf, size_t count,
                               ggml_type dtype, ggml_op_sum op, hipStream_t stream) {
    if (!initialized_) {
        fprintf(stderr, "GGMLRCCLComm::all_reduce: not initialized\n");
        return;
    }

    if (nDevices_ <= 1) {
        if (sendBuf != recvBuf) {
            if (stream) {
                HIP_CHECK(hipMemcpyAsync(recvBuf, sendBuf, count * ggml_type_sizeof(dtype), hipMemcpyDeviceToDevice, stream));
            } else {
                HIP_CHECK(hipMemcpy(recvBuf, sendBuf, count * ggml_type_sizeof(dtype), hipMemcpyDeviceToDevice));
            }
        }
        return;
    }

#if GGML_RCCL_NCCL_AVAILABLE
    if (use_rccl_) {
        all_reduce_rccl(sendBuf, recvBuf, count, dtype, op, stream);
        return;
    }
#endif

    all_reduce_ring(sendBuf, recvBuf, count, dtype, op, stream);
}

void GGMLRCCLComm::all_reduce_ring(const void* sendBuf, void* recvBuf, size_t count,
                                     ggml_type dtype, ggml_op_sum op, hipStream_t stream) {
    int curDevice;
    HIP_CHECK(hipGetDevice(&curDevice));

    size_t elemSize = ggml_type_sizeof(dtype);
    size_t totalSize = count * elemSize;

    if (sendBuf != recvBuf) {
        if (stream) {
            HIP_CHECK(hipMemcpyAsync(recvBuf, sendBuf, totalSize, hipMemcpyDeviceToDevice, stream));
            HIP_CHECK(hipStreamSynchronize(stream));
        } else {
            HIP_CHECK(hipMemcpy(recvBuf, sendBuf, totalSize, hipMemcpyDeviceToDevice));
        }
    }

    sync_all();

    if (host_buffers_[curDevice] == nullptr || totalSize > 64 * 1024 * 1024) {
        void* temp;
        HIP_CHECK(hipHostMalloc(&temp, totalSize, hipHostMallocDefault));

        for (int step = 0; step < nDevices_ - 1; step++) {
            int sender = (curDevice + step + 1) % nDevices_;

            barrier_wait(step);

            HIP_CHECK(hipMemcpy(temp, recvBuf, totalSize, hipMemcpyDeviceToHost));

            barrier_wait(step);

            void* sendData = temp;
            if (sender != curDevice && sender != ((curDevice + step) % nDevices_)) {
                void* senderBuf;
                HIP_CHECK(hipSetDevice(sender));
                HIP_CHECK(hipHostMalloc(&senderBuf, totalSize, hipHostMallocDefault));
                HIP_CHECK(hipMemcpy(senderBuf, recvBuf, totalSize, hipMemcpyDeviceToHost));
                HIP_CHECK(hipSetDevice(curDevice));
                sendData = senderBuf;
            }

            char* pRecv = (char*)recvBuf;
            char* pSend = (char*)sendData;
            size_t n = count;

            switch (dtype) {
                case GGML_TYPE_F32: {
                    float* dst = (float*)recvBuf;
                    float* src = (float*)sendData;
                    for (size_t i = 0; i < n; i++) {
                        dst[i] += src[i];
                    }
                    break;
                }
                case GGML_TYPE_F16: {
                    __half* dst = (__half*)recvBuf;
                    __half* src = (__half*)sendData;
                    for (size_t i = 0; i < n; i++) {
                        dst[i] = __hadd(dst[i], src[i]);
                    }
                    break;
                }
                default:
                    fprintf(stderr, "GGMLRCCLComm: unsupported dtype for ring reduce: %s\n",
                            ggml_type_name(dtype));
                    break;
            }

            if (sender != curDevice && sender != ((curDevice + step) % nDevices_)) {
                HIP_CHECK(hipSetDevice(sender));
                HIP_CHECK(hipHostFree(senderBuf));
                HIP_CHECK(hipSetDevice(curDevice));
            }
        }

        HIP_CHECK(hipHostFree(temp));
    } else {
        for (int step = 0; step < nDevices_ - 1; step++) {
            int sender = (curDevice + step + 1) % nDevices_;

            barrier_wait(step);

            HIP_CHECK(hipMemcpy(host_buffers_[curDevice], recvBuf, totalSize, hipMemcpyDeviceToHost));

            barrier_wait(step);

            char* pRecv = (char*)recvBuf;
            char* pSend = (char*)host_buffers_[sender];
            size_t n = count;

            switch (dtype) {
                case GGML_TYPE_F32: {
                    float* dst = (float*)recvBuf;
                    float* src = (float*)host_buffers_[sender];
                    for (size_t i = 0; i < n; i++) {
                        dst[i] += src[i];
                    }
                    break;
                }
                case GGML_TYPE_F16: {
                    __half* dst = (__half*)recvBuf;
                    __half* src = (__half*)host_buffers_[sender];
                    for (size_t i = 0; i < n; i++) {
                        dst[i] = __hadd(dst[i], src[i]);
                    }
                    break;
                }
                default:
                    break;
            }
        }
    }

    barrier_wait(nDevices_ - 1);
}

#if GGML_RCCL_NCCL_AVAILABLE
void GGMLRCCLComm::all_reduce_rccl(const void* sendBuf, void* recvBuf, size_t count,
                                     ggml_type dtype, ggml_op_sum op, hipStream_t stream) {
    int curDevice;
    HIP_CHECK(hipGetDevice(&curDevice));

    ncclDataType_t nccl_dtype;
    switch (dtype) {
        case GGML_TYPE_F32:  nccl_dtype = ncclFloat32; break;
        case GGML_TYPE_F16:  nccl_dtype = ncclFloat16; break;
        case GGML_TYPE_BF16: nccl_dtype = ncclBfloat16; break;
        default:             nccl_dtype = ncclFloat16; break;
    }

    ncclRedOp_t nccl_op = ncclSum;

    RCCL_CHECK(ncclAllReduce(sendBuf, recvBuf, count, nccl_dtype, nccl_op,
                              nccl_comms_[curDevice], stream));
}
#endif

void GGMLRCCLComm::all_gather(const void* sendBuf, void* recvBuf, size_t count,
                               ggml_type dtype, hipStream_t stream) {
    if (!initialized_) {
        fprintf(stderr, "GGMLRCCLComm::all_gather: not initialized\n");
        return;
    }

    if (nDevices_ <= 1) {
        if (sendBuf != recvBuf) {
            if (stream) {
                HIP_CHECK(hipMemcpyAsync(recvBuf, sendBuf, count * ggml_type_sizeof(dtype), hipMemcpyDeviceToDevice, stream));
            } else {
                HIP_CHECK(hipMemcpy(recvBuf, sendBuf, count * ggml_type_sizeof(dtype), hipMemcpyDeviceToDevice));
            }
        }
        return;
    }

    all_gather_ring(sendBuf, recvBuf, count, dtype, stream);
}

void GGMLRCCLComm::all_gather_ring(const void* sendBuf, void* recvBuf, size_t count,
                                    ggml_type dtype, hipStream_t stream) {
    int curDevice;
    HIP_CHECK(hipGetDevice(&curDevice));

    size_t elemSize = ggml_type_sizeof(dtype);
    size_t sliceSize = count * elemSize;

    if (stream) {
        HIP_CHECK(hipStreamSynchronize(stream));
    }

    for (int step = 0; step < nDevices_ - 1; step++) {
        int sender = (curDevice + step + 1) % nDevices_;
        int recv_offset = sender * count;

        char* recvBase = (char*)recvBuf + recv_offset * elemSize;

        barrier_wait(step);

        HIP_CHECK(hipMemcpy(host_buffers_[curDevice], (char*)sendBuf, sliceSize, hipMemcpyDeviceToHost));

        barrier_wait(step);

        HIP_CHECK(hipMemcpy(recvBase, host_buffers_[sender], sliceSize, hipMemcpyHostToDevice));
    }

    barrier_wait(nDevices_ - 1);

    char* recvBase = (char*)recvBuf + curDevice * count * elemSize;
    if (recvBase != sendBuf) {
        HIP_CHECK(hipMemcpy(recvBase, sendBuf, sliceSize, hipMemcpyDeviceToDevice));
    }
}

void GGMLRCCLComm::reduce_scatter(const void* sendBuf, void* recvBuf, size_t count,
                                    ggml_type dtype, ggml_op_sum op, hipStream_t stream) {
    if (!initialized_) {
        fprintf(stderr, "GGMLRCCLComm::reduce_scatter: not initialized\n");
        return;
    }

    size_t elemSize = ggml_type_sizeof(dtype);
    size_t totalSize = nDevices_ * count * elemSize;

    void* temp;
    HIP_CHECK(hipMalloc(&temp, totalSize));

    all_gather(sendBuf, temp, count, dtype, stream);

    if (stream) {
        HIP_CHECK(hipStreamSynchronize(stream));
    }

    int curDevice;
    HIP_CHECK(hipGetDevice(&curDevice));
    char* gathered = (char*)temp + curDevice * count * elemSize;

    if (stream) {
        HIP_CHECK(hipMemcpyAsync(recvBuf, gathered, count * elemSize, hipMemcpyDeviceToDevice, stream));
    } else {
        HIP_CHECK(hipMemcpy(recvBuf, gathered, count * elemSize, hipMemcpyDeviceToDevice));
    }

    HIP_CHECK(hipFree(temp));
}

void GGMLRCCLComm::broadcast(const void* sendBuf, void* recvBuf, size_t count,
                              ggml_type dtype, int root, hipStream_t stream) {
    if (!initialized_) {
        fprintf(stderr, "GGMLRCCLComm::broadcast: not initialized\n");
        return;
    }

    int curDevice;
    HIP_CHECK(hipGetDevice(&curDevice));

    if (curDevice == root) {
        size_t totalSize = count * ggml_type_sizeof(dtype);
        if (stream) {
            HIP_CHECK(hipMemcpyAsync(recvBuf, sendBuf, totalSize, hipMemcpyDeviceToDevice, stream));
        } else {
            HIP_CHECK(hipMemcpy(recvBuf, sendBuf, totalSize, hipMemcpyDeviceToDevice));
        }
    }

    sync_all();
}

#if GGML_RCCL_NCCL_AVAILABLE
ncclComm_t GGMLRCCLComm::get_nccl_comm(int deviceId) const {
    if (deviceId < 0 || deviceId >= nDevices_) {
        return nullptr;
    }
    return nccl_comms_[deviceId];
}
#endif
