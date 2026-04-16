#pragma once

#include "ggml.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_TP_MAX_WEIGHT_NAME 256

struct ggml_tp_weight_info {
    char name[GGML_TP_MAX_WEIGHT_NAME];
    int64_t ne[GGML_TP_MAX_DIMS];
    int n_dims;
    int type;
    size_t offset;
    size_t size;
};

struct ggml_tp_shard_info {
    struct ggml_tp_weight_info * weights;
    int n_weights;
    int n_devices;
    int tp_size;
};

struct ggml_tp_shard_info * ggml_tp_create_shard_info(
    struct ggml_tp_weight_info * weights,
    int n_weights,
    int tp_size
);

void ggml_tp_free_shard_info(struct ggml_tp_shard_info * info);

struct ggml_tp_weight_shard {
    char name[GGML_TP_MAX_WEIGHT_NAME];
    int device_id;
    int64_t ne[GGML_TP_MAX_DIMS];
    int n_dims;
    int type;
    void * data;
    size_t size;
};

struct ggml_tp_weight_shard * ggml_tp_get_weight_shard(
    struct ggml_tp_shard_info * shard_info,
    const char * weight_name,
    int rank
);

int ggml_tp_count_weight_shards(
    struct ggml_tp_shard_info * shard_info,
    const char * weight_name
);

void ggml_tp_split_weight_column(
    struct ggml_tensor * weight,
    struct ggml_tensor ** shards,
    int n_shards,
    int rank
);

void ggml_tp_split_weight_row(
    struct ggml_tensor * weight,
    struct ggml_tensor ** shards,
    int n_shards,
    int rank
);

void ggml_tp_gather_weight_shards(
    struct ggml_tensor * output,
    struct ggml_tensor ** shards,
    int n_shards
);

struct ggml_tp_linear_layer {
    struct ggml_tensor * weight;
    struct ggml_tensor * bias;

    bool is_column_parallel;
    bool is_row_parallel;
    bool needs_all_reduce;

    int64_t output_size_per_partition;
    int64_t input_size_per_partition;
};

struct ggml_tp_linear_layer * ggml_tp_create_column_linear(
    struct ggml_context * ctx,
    const char * name,
    int64_t input_size,
    int64_t output_size,
    int tp_size,
    int type
);

struct ggml_tp_linear_layer * ggml_tp_create_row_linear(
    struct ggml_context * ctx,
    const char * name,
    int64_t input_size,
    int64_t output_size,
    int tp_size,
    int type
);

void ggml_tp_free_linear_layer(struct ggml_tp_linear_layer * layer);

struct ggml_tensor * ggml_tp_forward_column_linear(
    struct ggml_context * ctx,
    struct ggml_tp_linear_layer * layer,
    struct ggml_tensor * input,
    int tp_rank
);

struct ggml_tensor * ggml_tp_forward_row_linear(
    struct ggml_context * ctx,
    struct ggml_tp_linear_layer * layer,
    struct ggml_tensor * input,
    int tp_rank,
    struct ggml_tp_comm * comm
);

#ifdef __cplusplus
}
#endif
