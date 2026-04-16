#include "ggml-tensor-parallel/ggml-tp-comm.h"
#include "ggml.h"
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <atomic>

struct ggml_tp_comm {
    int n_devices;
    int * device_ids;
    int tp_rank;
    int tp_size;

    hipStream_t * streams;

    bool use_nccl;
    bool initialized;

    float * host_send_buffers[8];
    float * host_recv_buffers[8];
    bool use_ring_allreduce;
};

static const char * ggml_tp_get_error(int err) {
    switch (err) {
        case 0:    return "Success";
        default:   return "Unknown error";
    }
}

#define GGML_TP_CHECK(call) \
    do { \
        int err = call; \
        if (err != 0) { \
            fprintf(stderr, "GGML TP ERROR at %s:%d: %s\n", \
                    __FILE__, __LINE__, ggml_tp_get_error(err)); \
        } \
    } while(0)

struct ggml_tp_comm * ggml_tp_comm_init(int n_devices, const int * device_ids) {
    struct ggml_tp_comm * comm = (struct ggml_tp_comm *)calloc(1, sizeof(struct ggml_tp_comm));
    if (!comm) {
        fprintf(stderr, "GGML TP: failed to allocate comm\n");
        return NULL;
    }

    comm->n_devices = n_devices;
    comm->device_ids = (int *)calloc(n_devices, sizeof(int));
    memcpy(comm->device_ids, device_ids, n_devices * sizeof(int));

    comm->tp_rank = 0;
    comm->tp_size = n_devices;

    comm->streams = (hipStream_t *)calloc(n_devices, sizeof(hipStream_t));
    for (int i = 0; i < n_devices; i++) {
        hipSetDevice(device_ids[i]);
        hipStreamCreate(&comm->streams[i]);
    }

    comm->use_nccl = false;
    comm->use_ring_allreduce = true;

    for (int i = 0; i < n_devices; i++) {
        hipSetDevice(device_ids[i]);
        hipHostMalloc(&comm->host_send_buffers[i], 1024 * 1024 * sizeof(float), hipHostMallocDefault);
        hipHostMalloc(&comm->host_recv_buffers[i], 1024 * 1024 * sizeof(float), hipHostMallocDefault);
    }

    comm->initialized = true;
    fprintf(stderr, "GGML TP: using Ring AllReduce via host memory\n");
    return comm;
}

void ggml_tp_comm_free(struct ggml_tp_comm * comm) {
    if (!comm) return;

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
        static std::atomic<int> barrier(0);
        int expected = barrier.load();
        while (!barrier.compare_exchange_weak(expected, (expected + 1) % comm->tp_size)) {
            expected = barrier.load();
        }
        while (barrier.load() != 0) {
            usleep(100);
        }
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

    hipSetDevice(device_id);
    hipMemcpyAsync(comm->host_send_buffers[device_id], send_buf, count * sizeof(float),
                   hipMemcpyDeviceToHost, comm->streams[device_id]);
    hipStreamSynchronize(comm->streams[device_id]);

    ggml_tp_comm_barrier(comm);

    for (int i = 0; i < comm->n_devices; i++) {
        for (int j = 0; j < count; j++) {
            comm->host_recv_buffers[i][j] += comm->host_send_buffers[i][j];
        }
    }

    ggml_tp_comm_barrier(comm);

    hipSetDevice(device_id);
    hipMemcpyAsync(recv_buf, comm->host_recv_buffers[device_id], count * sizeof(float),
                   hipMemcpyHostToDevice, comm->streams[device_id]);
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

    ggml_fp16_to_fp32_row(send_buf, comm->host_send_buffers[device_id], count);
    ggml_tp_comm_all_reduce_sum_f32(comm, comm->host_send_buffers[device_id],
                                     comm->host_recv_buffers[device_id], count, device_id);
    ggml_fp32_to_fp16_row(comm->host_recv_buffers[device_id], recv_buf, count);
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

    hipSetDevice(device_id);
    hipMemcpyAsync(comm->host_send_buffers[device_id], send_buf, count * sizeof(float),
                   hipMemcpyDeviceToHost, comm->streams[device_id]);
    hipStreamSynchronize(comm->streams[device_id]);

    ggml_tp_comm_barrier(comm);

    size_t offset = device_id * count;
    for (size_t i = 0; i < count; i++) {
        recv_buf[offset + i] = comm->host_send_buffers[device_id][i];
    }

    ggml_tp_comm_barrier(comm);

    hipSetDevice(device_id);
    hipMemcpyAsync(recv_buf, comm->host_recv_buffers[device_id], count * sizeof(float),
                   hipMemcpyHostToDevice, comm->streams[device_id]);
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

    if (comm->tp_rank == root) {
        hipSetDevice(comm->device_ids[root]);
        hipMemcpyAsync(comm->host_send_buffers[root], buffer, count * sizeof(float),
                       hipMemcpyDeviceToHost, comm->streams[root]);
        hipStreamSynchronize(comm->streams[root]);
    }

    ggml_tp_comm_barrier(comm);

    if (comm->tp_rank != root) {
        hipSetDevice(device_id);
        hipMemcpyAsync(buffer, comm->host_send_buffers[root], count * sizeof(float),
                       hipMemcpyHostToDevice, comm->streams[device_id]);
    }

    ggml_tp_comm_barrier(comm);
}
