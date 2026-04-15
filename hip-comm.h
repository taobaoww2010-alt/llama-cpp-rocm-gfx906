#pragma once

#include <hip/hip_runtime.h>

#include <cstddef>
#include <vector>
#include <memory>
#include <mutex>
#include <atomic>

struct ggml_tensor;

class HIPComm {
public:
    static HIPComm& instance();

    HIPComm(const HIPComm&) = delete;
    HIPComm& operator=(const HIPComm&) = delete;
    HIPComm(HIPComm&&) = delete;
    HIPComm& operator=(HIPComm&&) = delete;

    bool is_initialized() const { return initialized_; }
    int get_n_devices() const { return nDevices_; }
    bool can_access_peer(int src, int dst) const { return peer_access_[src * nDevices_ + dst]; }

    void init(int nDevices, const int* deviceIds);
    void finalize();

    void sync_device(int deviceId);
    void sync_all();

    void copy_peer(void* dst, int dstDevice, const void* src, int srcDevice, size_t size, hipStream_t stream = nullptr);
    void all_reduce_simple(const void* sendBuf, void* recvBuf, size_t count, hipStream_t stream = nullptr);

private:
    HIPComm() : initialized_(false), nDevices_(0) {}
    ~HIPComm();

    bool initialized_;
    int nDevices_;
    std::vector<int> devices_;
    std::vector<bool> peer_access_;
    std::mutex mutex_;
};

class HIPTimer {
public:
    HIPTimer() : start_time_(0), valid_(false) {}

    void start(hipStream_t stream = nullptr) {
        if (!valid_) {
            hipEventCreate(&start_event_).ignore();
            hipEventCreate(&stop_event_).ignore();
            valid_ = true;
        }
        hipEventRecord(start_event_, stream);
    }

    float stop(hipStream_t stream = nullptr) {
        if (!valid_) return 0.0f;
        hipEventRecord(stop_event_, stream);
        hipEventSynchronize(stop_event_);
        float ms = 0.0f;
        hipEventElapsedTime(&ms, start_event_, stop_event_);
        return ms;
    }

    ~HIPTimer() {
        if (valid_) {
            hipEventDestroy(start_event_).ignore();
            hipEventDestroy(stop_event_).ignore();
        }
    }

private:
    hipEvent_t start_event_;
    hipEvent_t stop_event_;
    bool valid_;
};

void hip_comm_all_reduce_impl(
    const void* sendBuf,
    void* recvBuf,
    size_t count,
    const int* devices,
    int nDevices,
    hipStream_t stream
);

void hip_comm_barrier_impl(
    const int* devices,
    int nDevices
);

void hip_comm_broadcast_impl(
    void* buf,
    size_t count,
    int root,
    const int* devices,
    int nDevices,
    hipStream_t stream
);
