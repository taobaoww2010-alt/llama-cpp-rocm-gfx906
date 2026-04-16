#include "ggml-tensor-parallel/ggml-tp-comm.h"
#include "ggml/ggml.h"
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#if GGML_USE_RCCL
#include <rccl.h>
#endif

struct ggml_tp_comm {
    int n_devices;
    int * device_ids;
    int tp_rank;
    int tp_size;

    hipStream_t * streams;

#if GGML_USE_RCCL
    ncclUniqueId unique_id;
    ncclComm_t * nccl_comms;
    bool use_nccl;
#endif

    bool initialized;

    float * host_send_buffers[GGML_TP_MAX_DEVICES];
    float * host_recv_buffers[GGML_TP_MAX_DEVICES];
    bool use_ring_allreduce;
};

static const char * ggml_tp_get_error(ncclResult_t err) {
    switch (err) {
        case ncclSuccess:          return "Success";
        case ncclUnhandledCudaError: return "Unhandled CUDA error";
        case ncclSystemError:       return "System error";
        case ncclInternalError:     return "Internal error";
        case ncclInvalidArgument:   return "Invalid argument";
        case ncclInvalidUsage:      return "Invalid usage";
        case ncclNumResults:        return "Num results";
        default:                    return "Unknown error";
    }
}

#define GGML_TP_CHECK(call) \
    do { \
        ncclResult_t err = call; \
        if (err != ncclSuccess) { \
            fprintf(stderr, "GGML TP ERROR at %s:%d: %s\n", \
                    __FILE__, __LINE__, ggml_tp_get_error(err)); \
        } \
    } while (0)

struct ggml_tp_comm * ggml_tp_comm_init(int n_devices, const int * device_ids) {
    struct ggml_tp_comm * comm = calloc(1, sizeof(struct ggml_tp_comm));
    if (!comm) {
        fprintf(stderr, "GGML TP: failed to allocate comm\n");
        return NULL;
    }

    comm->n_devices = n_devices;
    comm->device_ids = calloc(n_devices, sizeof(int));
    memcpy(comm->device_ids, device_ids, n_devices * sizeof(int));

    comm->tp_rank = 0;
    comm->tp_size = n_devices;

    comm->streams = calloc(n_devices, sizeof(hipStream_t));
    for (int i = 0; i < n_devices; i++) {
        hipSetDevice(device_ids[i]);
        hipStreamCreate(&comm->streams[i]);
    }

#if GGML_USE_RCCL
    comm->use_nccl = false;
    comm->nccl_comms = calloc(n_devices, sizeof(ncclComm_t));

    int can_access[GGML_TP_MAX_DEVICES][GGML_TP_MAX_DEVICES];
    bool p2p_available = true;

    for (int i = 0; i < n_devices; i++) {
        for (int j = 0; j < n_devices; j++) {
            if (i != j) {
                hipDeviceCanAccessPeer(&can_access[i][j], device_ids[i], device_ids[j]);
                if (!can_access[i][j]) {
                    p2p_available = false;
                }
            }
        }
    }

    fprintf(stderr, "GGML TP: initializing NCCL communicator...\n");

    NCCL_CHECK(ncclGetUniqueId(&comm->unique_id));

    for (int i = 0; i < n_devices; i++) {
        hipSetDevice(device_ids[i]);
        ncclResult_t result = ncclCommInitRank(&comm->nccl_comms[i], n_devices,
                                               comm->unique_id, i);
        if (result != ncclSuccess) {
            fprintf(stderr, "GGML TP: NCCL init failed for device %d: %s\n",
                    device_ids[i], ncclGetErrorString(result));
            comm->use_nccl = false;
            break;
        }
        comm->use_nccl = true;
    }

    if (comm->use_nccl) {
        fprintf(stderr, "GGML TP: NCCL initialized successfully\n");
    } else {
        for (int i = 0; i < n_devices; i++) {
            if (comm->nccl_comms[i]) {
                ncclCommDestroy(comm->nccl_comms[i]);
            }
        }
        free(comm->nccl_comms);
        comm->nccl_comms = NULL;
    }
#endif

    if (!comm->use_nccl) {
        fprintf(stderr, "GGML TP: using Ring AllReduce via host memory\n");
        comm->use_ring_allreduce = true;

        for (int i = 0; i < n_devices; i++) {
            hipSetDevice(device_ids[i]);
            hipHostMalloc(&comm->host_send_buffers[i], GGML_TP_MAX_SIZE * sizeof(float), hipHostMallocDefault);
            hipHostMalloc(&comm->host_recv_buffers[i], GGML_TP_MAX_SIZE * sizeof(float), hipHostMallocDefault);
        }
    }

    comm->initialized = true;
    return comm;
}

void ggml_tp_comm_free(struct ggml_tp_comm * comm) {
    if (!comm) return;

#if GGML_USE_RCCL
    if (comm->nccl_comms) {
        for (int i = 0; i < comm->n_devices; i++) {
            if (comm->nccl_comms[i]) {
                ncclCommDestroy(comm->nccl_comms[i]);
            }
        }
        free(comm->nccl_comms);
    }
#endif

    for (int i = 0; i < comm->n_devices; i++) {
        hipSetDevice(comm->device_ids[i]);
        hipStreamDestroy(comm->streams[i]);
        if (comm->host_send_buffers[i]) {
            hipHostFree(comm->host_send_buffers[i]);
        }
        if (comm->host_recv_buffers[i]) {
            hipHostFree(comm->host_recv_buffers[i]);
        }
    }

    free(comm->device_ids);
    free(comm->streams);
    free(comm);
}

int ggml_tp_comm_get_rank(struct ggml_tp_comm * comm) {
    return comm ? comm->tp_rank : 0;
}

int ggml_tp_comm_get_size(struct ggml_tp_comm * comm) {
    return comm ? comm->tp_size : 1;
}

bool ggml_tp_comm_is_initialized(struct ggml_tp_comm * comm) {
    return comm && comm->initialized;
}

void ggml_tp_comm_barrier(struct ggml_tp_comm * comm) {
    for (int i = 0; i < comm->n_devices; i++) {
        hipSetDevice(comm->device_ids[i]);
        hipStreamSynchronize(comm->streams[i]);
    }

    if (comm->tp_size > 1) {
        volatile int barrier = 0;
        while (__atomic_load_n(&barrier, __ATOMIC_SEQ_CST) != comm->tp_rank) {
            usleep(100);
        }
        __atomic_store_n(&barrier, (barrier + 1) % comm->tp_size, __ATOMIC_SEQ_CST);
    }
}

void ggml_tp_comm_sync_device(struct ggml_tp_comm * comm, int device_id) {
    hipSetDevice(device_id);
    hipStreamSynchronize(comm->streams[device_id]);
}

void ggml_tp_comm_sync_all(struct ggml_tp_comm * comm) {
    for (int i = 0; i < comm->n_devices; i++) {
        ggml_tp_comm_sync_device(comm, i);
    }
}

static void ring_allreduce_f32(
    struct ggml_tp_comm * comm,
    float * send_buf,
    float * recv_buf,
    size_t count
) {
    int n = comm->n_devices;
    int rank = comm->tp_rank;

    int left = (rank - 1 + n) % n;
    int right = (rank + 1) % n;

    for (int step = 0; step < n - 1; step++) {
        int src_rank = (rank - step - 1 + n) % n;
        int dst_rank = (rank + step + 1) % n;

        float * src_host = comm->host_send_buffers[src_rank];
        float * dst_host = comm->host_recv_buffers[dst_rank];

        hipSetDevice(comm->device_ids[rank]);
        hipMemcpyAsync(src_host, send_buf, count * sizeof(float),
                       hipMemcpyDeviceToHost, comm->streams[rank]);

        int my_src_device = comm->device_ids[src_rank];
        hipSetDevice(my_src_device);
        hipMemcpyAsync(dst_host, src_host, count * sizeof(float),
                       hipMemcpyHostToDevice, comm->streams[my_src_device]);

        for (int i = 0; i < n; i++) {
            hipSetDevice(comm->device_ids[i]);
            hipStreamSynchronize(comm->streams[i]);
        }

        hipSetDevice(comm->device_ids[rank]);
        hipMemcpyAsync(recv_buf, dst_host, count * sizeof(float),
                       hipMemcpyHostToDevice, comm->streams[rank]);
    }
}

void ggml_tp_comm_all_reduce_sum_f32(
    struct ggml_tp_comm * comm,
    float * send_buf,
    float * recv_buf,
    size_t count,
    int device_id
) {
    if (comm->tp_size <= 1) {
        if (send_buf != recv_buf) {
            memcpy(recv_buf, send_buf, count * sizeof(float));
        }
        return;
    }

#if GGML_USE_RCCL
    if (comm->use_nccl) {
        hipSetDevice(device_id);
        GGML_TP_CHECK(ncclAllReduce(
            send_buf, recv_buf, count, ncclFloat32, ncclSum,
            comm->nccl_comms[comm->tp_rank], comm->streams[device_id]
        ));
        return;
    }
#endif

    ring_allreduce_f32(comm, send_buf, recv_buf, count);
}

void ggml_tp_comm_all_reduce_sum_f16(
    struct ggml_tp_comm * comm,
    ggml_fp16_t * send_buf,
    ggml_fp16_t * recv_buf,
    size_t count,
    int device_id
) {
    if (comm->tp_size <= 1) {
        if (send_buf != recv_buf) {
            memcpy(recv_buf, send_buf, count * sizeof(ggml_fp16_t));
        }
        return;
    }

#if GGML_USE_RCCL
    if (comm->use_nccl) {
        hipSetDevice(device_id);
        GGML_TP_CHECK(ncclAllReduce(
            send_buf, recv_buf, count, ncclFloat16, ncclSum,
            comm->nccl_comms[comm->tp_rank], comm->streams[device_id]
        ));
        return;
    }
#endif

    float * send_f32 = comm->host_send_buffers[device_id];
    float * recv_f32 = comm->host_recv_buffers[device_id];

    ggml_fp16_to_fp32_server(send_buf, send_f32, count);

    ring_allreduce_f32(comm, send_f32, recv_f32, count);

    ggml_fp32_to_fp16_server(recv_f32, recv_buf, count);
}

void ggml_tp_comm_all_gather_f32(
    struct ggml_tp_comm * comm,
    const float * send_buf,
    float * recv_buf,
    size_t count,
    int device_id
) {
    if (comm->tp_size <= 1) {
        memcpy(recv_buf, send_buf, count * sizeof(float));
        return;
    }

#if GGML_USE_RCCL
    if (comm->use_nccl) {
        hipSetDevice(device_id);
        GGML_TP_CHECK(ncclAllGather(
            send_buf, recv_buf, count, ncclFloat32,
            comm->nccl_comms[comm->tp_rank], comm->streams[device_id]
        ));
        return;
    }
#endif

    int n = comm->tp_size;
    size_t chunk_size = count * sizeof(float);

    for (int i = 0; i < n; i++) {
        int src_rank = (comm->tp_rank - i + n) % n;
        int dst_offset = src_rank * count;

        hipSetDevice(comm->device_ids[device_id]);
        hipMemcpyAsync(recv_buf + dst_offset, send_buf, chunk_size,
                       hipMemcpyDeviceToHost, comm->streams[device_id]);
    }

    hipSetDevice(device_id);
    hipStreamSynchronize(comm->streams[device_id]);
}

void ggml_tp_comm_reduce_scatter_f32(
    struct ggml_tp_comm * comm,
    const float * send_buf,
    float * recv_buf,
    size_t count,
    int device_id
) {
    if (comm->tp_size <= 1) {
        memcpy(recv_buf, send_buf, count * sizeof(float));
        return;
    }

#if GGML_USE_RCCL
    if (comm->use_nccl) {
        hipSetDevice(device_id);
        GGML_TP_CHECK(ncclReduceScatter(
            send_buf, recv_buf, count, ncclFloat32, ncclSum,
            comm->nccl_comms[comm->tp_rank], comm->streams[device_id]
        ));
        return;
    }
#endif

    fprintf(stderr, "GGML TP: reduce_scatter not implemented for Ring mode\n");
}

void ggml_tp_comm_broadcast_f32(
    struct ggml_tp_comm * comm,
    float * buffer,
    size_t count,
    int root,
    int device_id
) {
    if (comm->tp_size <= 1) {
        return;
    }

#if GGML_USE_RCCL
    if (comm->use_nccl) {
        hipSetDevice(device_id);
        GGML_TP_CHECK(ncclBroadcast(
            buffer, buffer, count, ncclFloat32, root,
            comm->nccl_comms[device_id], comm->streams[device_id]
        ));
        return;
    }
#endif

    if (comm->tp_rank == root) {
        for (int i = 0; i < comm->n_devices; i++) {
            if (i != root) {
                hipSetDevice(comm->device_ids[root]);
                hipMemcpyAsync(comm->host_send_buffers[i], buffer,
                              count * sizeof(float), hipMemcpyDeviceToHost,
                              comm->streams[root]);
            }
        }
    }

    ggml_tp_comm_barrier(comm);

    if (comm->tp_rank != root) {
        hipSetDevice(device_id);
        hipMemcpyAsync(buffer, comm->host_send_buffers[comm->tp_rank],
                      count * sizeof(float), hipMemcpyHostToDevice,
                      comm->streams[device_id]);
    }
}
