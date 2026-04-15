#include "hip-comm.h"

#include "../ggml-impl.h"

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <algorithm>

#define HIP_CHECK(call)                                                        \
    do {                                                                       \
        hipError_t err = call;                                                  \
        if (err != hipSuccess) {                                               \
            fprintf(stderr, "HIP error at %s:%d: %s\n",                         \
                    __FILE__, __LINE__, hipGetErrorString(err));                \
        }                                                                      \
    } while (0)

HIPComm::~HIPComm() {
    if (initialized_) {
        finalize();
    }
}

HIPComm& HIPComm::instance() {
    static HIPComm instance_;
    return instance_;
}

void HIPComm::init(int nDevices, const int* deviceIds) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (initialized_) {
        fprintf(stderr, "HIPComm already initialized\n");
        return;
    }

    if (nDevices < 1) {
        fprintf(stderr, "HIPComm: invalid nDevices=%d\n", nDevices);
        return;
    }

    nDevices_ = nDevices;
    devices_.resize(nDevices);
    peer_access_.resize(nDevices * nDevices, false);

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
                        fprintf(stderr, "HIPComm: P2P enabled GPU %d -> GPU %d\n", i, j);
                    }
                }
            }
        }
    }

    initialized_ = true;
    fprintf(stderr, "HIPComm: initialized %d devices\n", nDevices);
}

void HIPComm::finalize() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!initialized_) {
        return;
    }

    for (int i = 0; i < nDevices_; i++) {
        for (int j = 0; j < nDevices_; j++) {
            if (peer_access_[i * nDevices_ + j]) {
                HIP_CHECK(hipSetDevice(devices_[i]));
                HIP_CHECK(hipDeviceDisablePeerAccess(devices_[j]));
            }
        }
    }

    devices_.clear();
    peer_access_.clear();
    initialized_ = false;
    nDevices_ = 0;

    fprintf(stderr, "HIPComm: finalized\n");
}

void HIPComm::sync_device(int deviceId) {
    HIP_CHECK(hipSetDevice(deviceId));
    HIP_CHECK(hipDeviceSynchronize());
}

void HIPComm::sync_all() {
    for (int i = 0; i < nDevices_; i++) {
        sync_device(i);
    }
}

void HIPComm::copy_peer(void* dst, int dstDevice, const void* src, int srcDevice, size_t size, hipStream_t stream) {
    if (dstDevice == srcDevice) {
        if (stream) {
            HIP_CHECK(hipMemcpyAsync(dst, src, size, hipMemcpyDeviceToDevice, stream));
        } else {
            HIP_CHECK(hipMemcpy(dst, src, size, hipMemcpyDeviceToDevice));
        }
        return;
    }

    if (peer_access_[srcDevice * nDevices_ + dstDevice]) {
        if (stream) {
            HIP_CHECK(hipMemcpyAsync(dst, src, size, hipMemcpyDeviceToDevice, stream));
        } else {
            HIP_CHECK(hipMemcpy(dst, src, size, hipMemcpyDeviceToDevice));
        }
    } else {
        void* temp = nullptr;
        HIP_CHECK(hipMalloc(&temp, size));
        if (stream) {
            HIP_CHECK(hipMemcpyAsync(temp, src, size, hipMemcpyDeviceToDevice, stream));
            HIP_CHECK(hipStreamSynchronize(stream));
        } else {
            HIP_CHECK(hipMemcpy(temp, src, size, hipMemcpyDeviceToDevice));
        }
        HIP_CHECK(hipSetDevice(dstDevice));
        if (stream) {
            HIP_CHECK(hipMemcpyAsync(dst, temp, size, hipMemcpyDeviceToHost, stream));
        } else {
            HIP_CHECK(hipMemcpy(dst, temp, size, hipMemcpyDeviceToHost));
        }
        HIP_CHECK(hipFree(temp));
    }
}

namespace {

struct ReductionOp {
    template<typename T>
    __device__ __host__ T operator()(T a, T b) const {
        return a + b;
    }
};

template<typename T>
__global__ void reduce_kernel(T* data, int n, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (idx >= n) return;

    T val = data[idx];
    for (int s = stride; s > 0; s >>= 1) {
        if (tid < s && idx + s < n) {
            val = val + data[idx + s];
        }
        __syncthreads();
        if (tid < s && idx + s < n) {
            data[idx] = val;
        }
        __syncthreads();
    }
}

}

void HIPComm::all_reduce_simple(const void* sendBuf, void* recvBuf, size_t count, hipStream_t stream) {
    if (nDevices_ <= 1) {
        if (sendBuf != recvBuf && stream) {
            hipMemcpyAsync(recvBuf, sendBuf, count * sizeof(float), hipMemcpyDeviceToDevice, stream);
        } else if (sendBuf != recvBuf) {
            hipMemcpy(recvBuf, sendBuf, count * sizeof(float), hipMemcpyDeviceToDevice);
        }
        return;
    }

    int curDevice;
    HIP_CHECK(hipGetDevice(&curDevice));

    if (sendBuf != recvBuf) {
        if (stream) {
            HIP_CHECK(hipMemcpyAsync(recvBuf, sendBuf, count * sizeof(float), hipMemcpyDeviceToDevice, stream));
        } else {
            HIP_CHECK(hipMemcpy(recvBuf, sendBuf, count * sizeof(float), hipMemcpyDeviceToDevice));
        }
    }

    if (stream) {
        HIP_CHECK(hipStreamSynchronize(stream));
    }

    sync_all();

    const int nGPUs = nDevices_;
    const size_t chunkSize = count / nGPUs;
    const size_t remainder = count % nGPUs;

    std::vector<void*> recvBuffers(nGPUs);
    std::vector<hipEvent_t> events(nGPUs);

    for (int i = 0; i < nGPUs; i++) {
        HIP_CHECK(hipSetDevice(devices_[i]));
        HIP_CHECK(hipEventCreate(&events[i]));

        size_t offset = i * chunkSize;
        size_t size = chunkSize + (i < (int)remainder ? 1 : 0);
        size_t byteOffset = offset * sizeof(float);

        recvBuffers[i] = (char*)recvBuf + byteOffset;
    }

    for (int step = 1; step < nGPUs; step++) {
        for (int i = 0; i < nGPUs; i++) {
            int src = (i + step) % nGPUs;

            if (peer_access_[i * nDevices_ + src]) {
                HIP_CHECK(hipSetDevice(devices_[i]));
                HIP_CHECK(hipMemcpyAsync(recvBuffers[i], recvBuffers[src],
                                          chunkSize * sizeof(float),
                                          hipMemcpyDeviceToDevice, stream));
            } else {
                void* tempHost = malloc(chunkSize * sizeof(float));

                HIP_CHECK(hipSetDevice(devices_[src]));
                if (stream) {
                    HIP_CHECK(hipMemcpyAsync(tempHost, recvBuffers[src],
                                              chunkSize * sizeof(float),
                                              hipMemcpyDeviceToHost, stream));
                    HIP_CHECK(hipStreamSynchronize(stream));
                } else {
                    HIP_CHECK(hipMemcpy(tempHost, recvBuffers[src],
                                          chunkSize * sizeof(float),
                                          hipMemcpyDeviceToHost));
                }

                HIP_CHECK(hipSetDevice(devices_[i]));
                if (stream) {
                    HIP_CHECK(hipMemcpyAsync(recvBuffers[i], tempHost,
                                              chunkSize * sizeof(float),
                                              hipMemcpyHostToDevice, stream));
                } else {
                    HIP_CHECK(hipMemcpy(recvBuffers[i], tempHost,
                                          chunkSize * sizeof(float),
                                          hipMemcpyHostToDevice));
                }

                free(tempHost);
            }
        }

        if (stream) {
            HIP_CHECK(hipStreamSynchronize(stream));
        }
        sync_all();
    }

    for (int i = 0; i < nGPUs; i++) {
        HIP_CHECK(hipEventDestroy(events[i]));
    }

    HIP_CHECK(hipSetDevice(curDevice));
}

void hip_comm_all_reduce_impl(
    const void* sendBuf,
    void* recvBuf,
    size_t count,
    const int* devices,
    int nDevices,
    hipStream_t stream
) {
    HIPComm::instance().all_reduce_simple(sendBuf, recvBuf, count, stream);
}

void hip_comm_barrier_impl(
    const int* devices,
    int nDevices
) {
    for (int i = 0; i < nDevices; i++) {
        hipSetDevice(devices[i]);
        hipDeviceSynchronize();
    }
}

void hip_comm_broadcast_impl(
    void* buf,
    size_t count,
    int root,
    const int* devices,
    int nDevices,
    hipStream_t stream
) {
    if (nDevices <= 1) return;

    int curDevice;
    hipGetDevice(&curDevice);

    for (int i = 0; i < nDevices; i++) {
        if (i == root) continue;

        hipSetDevice(devices[i]);
        if (stream) {
            hipMemcpyAsync(buf, buf, count * sizeof(float), hipMemcpyDeviceToDevice, stream);
        } else {
            hipMemcpy(buf, buf, count * sizeof(float), hipMemcpyDeviceToDevice);
        }
    }

    hipSetDevice(devices[root]);

    if (stream) {
        hipStreamSynchronize(stream);
    }

    hipSetDevice(curDevice);
}
