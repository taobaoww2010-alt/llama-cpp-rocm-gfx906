#pragma once

#include "ggml/ggml.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_tp_comm;

struct ggml_tp_comm * ggml_tp_comm_init(
    int n_devices,
    const int * device_ids
);

void ggml_tp_comm_free(struct ggml_tp_comm * comm);

int ggml_tp_comm_get_rank(struct ggml_tp_comm * comm);

int ggml_tp_comm_get_size(struct ggml_tp_comm * comm);

bool ggml_tp_comm_is_initialized(struct ggml_tp_comm * comm);

void ggml_tp_comm_barrier(struct ggml_tp_comm * comm);

void ggml_tp_comm_sync_device(struct ggml_tp_comm * comm, int device_id);

void ggml_tp_comm_sync_all(struct ggml_tp_comm * comm);

void ggml_tp_comm_all_reduce_sum_f32(
    struct ggml_tp_comm * comm,
    float * send_buf,
    float * recv_buf,
    size_t count,
    int device_id
);

void ggml_tp_comm_all_reduce_sum_f16(
    struct ggml_tp_comm * comm,
    ggml_fp16_t * send_buf,
    ggml_fp16_t * recv_buf,
    size_t count,
    int device_id
);

void ggml_tp_comm_all_reduce_sum_q8_0(
    struct ggml_tp_comm * comm,
    void * send_buf,
    void * recv_buf,
    size_t count,
    int device_id
);

void ggml_tp_comm_all_gather_f32(
    struct ggml_tp_comm * comm,
    const float * send_buf,
    float * recv_buf,
    size_t count,
    int device_id
);

void ggml_tp_comm_all_gather_f16(
    struct ggml_tp_comm * comm,
    const ggml_fp16_t * send_buf,
    ggml_fp16_t * recv_buf,
    size_t count,
    int device_id
);

void ggml_tp_comm_reduce_scatter_f32(
    struct ggml_tp_comm * comm,
    const float * send_buf,
    float * recv_buf,
    size_t count,
    int device_id
);

void ggml_tp_comm_broadcast_f32(
    struct ggml_tp_comm * comm,
    float * buffer,
    size_t count,
    int root,
    int device_id
);

#ifdef __cplusplus
}
#endif
