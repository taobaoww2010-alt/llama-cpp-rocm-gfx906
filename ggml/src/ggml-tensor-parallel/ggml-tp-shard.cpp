#include "ggml-tensor-parallel/ggml-tp-shard.h"
#include "ggml.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct ggml_tp_shard_info * ggml_tp_create_shard_info(
    struct ggml_tp_weight_info * weights,
    int n_weights,
    int tp_size
) {
    struct ggml_tp_shard_info * info = calloc(1, sizeof(struct ggml_tp_shard_info));
    if (!info) return NULL;

    info->weights = weights;
    info->n_weights = n_weights;
    info->tp_size = tp_size;
    info->n_devices = tp_size;

    return info;
}

void ggml_tp_free_shard_info(struct ggml_tp_shard_info * info) {
    if (!info) return;
    if (info->weights) {
        free(info->weights);
    }
    free(info);
}

struct ggml_tp_weight_shard * ggml_tp_get_weight_shard(
    struct ggml_tp_shard_info * shard_info,
    const char * weight_name,
    int rank
) {
    if (!shard_info || !weight_name) return NULL;

    for (int i = 0; i < shard_info->n_weights; i++) {
        if (strcmp(shard_info->weights[i].name, weight_name) == 0) {
            struct ggml_tp_weight_shard * shard = calloc(1, sizeof(struct ggml_tp_weight_shard));
            if (!shard) return NULL;

            strncpy(shard->name, weight_name, GGML_TP_MAX_WEIGHT_NAME - 1);
            shard->device_id = rank;
            shard->type = shard_info->weights[i].type;
            shard->n_dims = shard_info->weights[i].n_dims;

            int64_t total_elements = 1;
            for (int d = 0; d < shard->n_dims; d++) {
                shard->ne[d] = shard_info->weights[i].ne[d];
                total_elements *= shard->ne[d];
            }

            if (shard->n_dims >= 2) {
                shard->ne[0] = shard_info->weights[i].ne[0] / shard_info->tp_size;
            }

            shard->size = total_elements * ggml_type_size(shard->type);

            return shard;
        }
    }

    return NULL;
}

int ggml_tp_count_weight_shards(
    struct ggml_tp_shard_info * shard_info,
    const char * weight_name
) {
    for (int i = 0; i < shard_info->n_weights; i++) {
        if (strcmp(shard_info->weights[i].name, weight_name) == 0) {
            return shard_info->tp_size;
        }
    }
    return 1;
}

void ggml_tp_split_weight_column(
    struct ggml_tensor * weight,
    struct ggml_tensor ** shards,
    int n_shards,
    int rank
) {
    if (!weight || !shards || n_shards <= 0) return;

    if (weight->n_dims < 2) {
        shards[rank] = weight;
        return;
    }

    int64_t ne0_per_shard = weight->ne[0] / n_shards;
    int64_t offset = rank * ne0_per_shard * ggml_type_size(weight->type) * weight->ne[1];

    shards[rank] = ggml_view_2d(
        weight->ctx,
        weight,
        ne0_per_shard,
        weight->ne[1],
        weight->nb[1],
        offset
    );
}

void ggml_tp_split_weight_row(
    struct ggml_tensor * weight,
    struct ggml_tensor ** shards,
    int n_shards,
    int rank
) {
    if (!weight || !shards || n_shards <= 0) return;

    if (weight->n_dims < 2) {
        shards[rank] = weight;
        return;
    }

    int64_t ne1_per_shard = weight->ne[1] / n_shards;
    int64_t offset = rank * ne1_per_shard * ggml_type_size(weight->type);

    shards[rank] = ggml_view_2d(
        weight->ctx,
        weight,
        weight->ne[0],
        ne1_per_shard,
        weight->nb[2],
        offset
    );
}

void ggml_tp_gather_weight_shards(
    struct ggml_tensor * output,
    struct ggml_tensor ** shards,
    int n_shards
) {
    if (!output || !shards || n_shards <= 0) return;

}

struct ggml_tp_linear_layer * ggml_tp_create_column_linear(
    struct ggml_context * ctx,
    const char * name,
    int64_t input_size,
    int64_t output_size,
    int tp_size,
    int type
) {
    struct ggml_tp_linear_layer * layer = calloc(1, sizeof(struct ggml_tp_linear_layer));
    if (!layer) return NULL;

    layer->is_column_parallel = true;
    layer->is_row_parallel = false;
    layer->needs_all_reduce = false;

    layer->output_size_per_partition = output_size / tp_size;
    layer->input_size_per_partition = input_size;

    layer->weight = ggml_new_tensor_2d(ctx, type, input_size, layer->output_size_per_partition);

    return layer;
}

struct ggml_tp_linear_layer * ggml_tp_create_row_linear(
    struct ggml_context * ctx,
    const char * name,
    int64_t input_size,
    int64_t output_size,
    int tp_size,
    int type
) {
    struct ggml_tp_linear_layer * layer = calloc(1, sizeof(struct ggml_tp_linear_layer));
    if (!layer) return NULL;

    layer->is_column_parallel = false;
    layer->is_row_parallel = true;
    layer->needs_all_reduce = true;

    layer->output_size_per_partition = output_size;
    layer->input_size_per_partition = input_size / tp_size;

    layer->weight = ggml_new_tensor_2d(ctx, type, layer->input_size_per_partition, output_size);

    return layer;
}

void ggml_tp_free_linear_layer(struct ggml_tp_linear_layer * layer) {
    if (!layer) return;
    free(layer);
}

struct ggml_tensor * ggml_tp_forward_column_linear(
    struct ggml_context * ctx,
    struct ggml_tp_linear_layer * layer,
    struct ggml_tensor * input,
    int tp_rank
) {
    if (!layer || !input) return NULL;

    struct ggml_tensor * result = ggml_mul_mat(ctx, layer->weight, input);

    return result;
}

struct ggml_tensor * ggml_tp_forward_row_linear(
    struct ggml_context * ctx,
    struct ggml_tp_linear_layer * layer,
    struct ggml_tensor * input,
    int tp_rank,
    struct ggml_tp_comm * comm
) {
    if (!layer || !input) return NULL;

    struct ggml_tensor * result = ggml_mul_mat(ctx, layer->weight, input);

    return result;
}
