#pragma once

#include <hip/hip_runtime.h>

#include "../ggml.h"

#include <cstddef>
#include <vector>
#include <memory>
#include <mutex>
#include <condition_variable>

#define GGML_RCCL_NCCL_AVAILABLE 0

#if GGML_RCCL_NCCL_AVAILABLE
#include <rccl.h>
#endif

class GGMLRCCLComm {
public:
    static GGMLRCCLComm& instance();

    GGMLRCCLComm(const GGMLRCCLComm&) = delete;
    GGMLRCCLComm& operator=(const GGMLRCCLComm&) = delete;
    GGMLRCCLComm(GGMLRCCLComm&&) = delete;
    GGMLRCCLComm& operator=(GGMLRCCLComm&&) = delete;

    bool is_initialized() const { return initialized_; }
    int get_n_devices() const { return nDevices_; }
    int get_rank() const { return rank_; }
    bool using_rccl() const { return use_rccl_; }
    bool can_p2p(int src, int dst) const { return peer_access_[src * nDevices_ + dst]; }

    void init(int nDevices, const int* deviceIds);
    void finalize();

    void all_reduce(const void* sendBuf, void* recvBuf, size_t count,
                    ggml_type dtype, ggml_op_sum op, hipStream_t stream);

    void all_gather(const void* sendBuf, void* recvBuf, size_t count,
                    ggml_type dtype, hipStream_t stream);

    void reduce_scatter(const void* sendBuf, void* recvBuf, size_t count,
                        ggml_type dtype, ggml_op_sum op, hipStream_t stream);

    void broadcast(const void* sendBuf, void* recvBuf, size_t count,
                   ggml_type dtype, int root, hipStream_t stream);

    void sync_device(int deviceId);
    void sync_all();

#if GGML_RCCL_NCCL_AVAILABLE
    ncclComm_t get_nccl_comm(int deviceId) const;
#endif

private:
    GGMLRCCLComm();
    ~GGMLRCCLComm();

    void barrier_wait(int step);

    void all_reduce_rccl(const void* sendBuf, void* recvBuf, size_t count,
                         ggml_type dtype, ggml_op_sum op, hipStream_t stream);
    void all_reduce_ring(const void* sendBuf, void* recvBuf, size_t count,
                          ggml_type dtype, ggml_op_sum op, hipStream_t stream);

    void all_gather_ring(const void* sendBuf, void* recvBuf, size_t count,
                          ggml_type dtype, hipStream_t stream);

    bool initialized_;
    bool use_rccl_;
    int nDevices_;
    int rank_;
    std::vector<int> devices_;
    std::vector<bool> peer_access_;
    std::vector<void*> host_buffers_;
    std::vector<std::mutex> step_mutex_;
    std::vector<std::condition_variable> step_cv_;
    std::vector<int> step_counts_;
    std::vector<int> step_generations_;
    std::mutex mutex_;

#if GGML_RCCL_NCCL_AVAILABLE
    std::vector<ncclComm_t> nccl_comms_;
    ncclUniqueId uniqueId_;
#endif
};

inline ggml_type ggml_backend_buffer_type_to_acc_dtype(ggml_backend_buffer_type_t buft) {
    if (buft == nullptr) return GGML_TYPE_F32;

    const char* name = ggml_backend_buffer_type_get_name(buft);
    if (strcmp(name, "CUDA") == 0 || strcmp(name, "HIP") == 0) {
        return GGML_TYPE_F16;
    }
    return GGML_TYPE_F32;
}
