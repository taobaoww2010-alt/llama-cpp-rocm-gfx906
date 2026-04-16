#include "ggml-cuda-tp.h"
#include "ggml-cuda.h"
#include "ggml-tensor-parallel.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct ggml_cuda_tp_context {
    struct ggml_tensor_parallel * tp;

    rocblas_handle * rocblas_handles;
    hipStream_t * streams;

    void * workspace;
    size_t workspace_size;
};

static struct ggml_cuda_tp_context * cuda_tp_ctx = NULL;

void ggml_cuda_tp_init(
    struct ggml_tensor_parallel * tp,
    int n_devices,
    const int * device_ids
) {
    if (cuda_tp_ctx) {
        ggml_cuda_tp_free(tp);
    }

    cuda_tp_ctx = (struct ggml_cuda_tp_context *) calloc(1, sizeof(struct ggml_cuda_tp_context));
    if (!cuda_tp_ctx) {
        fprintf(stderr, "GGML CUDA TP: failed to allocate context\n");
        return;
    }

    cuda_tp_ctx->tp = tp;

    cuda_tp_ctx->rocblas_handles = (rocblas_handle *) calloc(n_devices, sizeof(rocblas_handle));
    cuda_tp_ctx->streams = (hipStream_t *) calloc(n_devices, sizeof(hipStream_t));

    for (int i = 0; i < n_devices; i++) {
        hipSetDevice(device_ids[i]);

        rocblas_create_handle(&cuda_tp_ctx->rocblas_handles[i]);
        hipStreamCreate(&cuda_tp_ctx->streams[i]);
    }

    cuda_tp_ctx->workspace_size = 256 * 1024 * 1024;
    for (int i = 0; i < n_devices; i++) {
        hipSetDevice(device_ids[i]);
        hipMalloc(&cuda_tp_ctx->workspace, cuda_tp_ctx->workspace_size);
    }

    ggml_tp_init(tp, n_devices, device_ids);

    fprintf(stderr, "GGML CUDA TP: initialized tensor parallel context\n");
}

void ggml_cuda_tp_free(struct ggml_tensor_parallel * tp) {
    if (!cuda_tp_ctx) return;

    if (cuda_tp_ctx->workspace) {
        for (int i = 0; i < cuda_tp_ctx->tp->n_devices; i++) {
            hipSetDevice(cuda_tp_ctx->tp->device_ids[i]);
            hipFree(cuda_tp_ctx->workspace);
        }
    }

    for (int i = 0; i < cuda_tp_ctx->tp->n_devices; i++) {
        hipSetDevice(cuda_tp_ctx->tp->device_ids[i]);
        rocblas_destroy_handle(cuda_tp_ctx->rocblas_handles[i]);
        hipStreamDestroy(cuda_tp_ctx->streams[i]);
    }

    free(cuda_tp_ctx->rocblas_handles);
    free(cuda_tp_ctx->streams);

    if (cuda_tp_ctx->tp) {
        ggml_tp_free(cuda_tp_ctx->tp);
    }

    free(cuda_tp_ctx);
    cuda_tp_ctx = NULL;
}

struct ggml_tensor * ggml_cuda_tp_mul_mat(
    struct ggml_tensor_parallel * tp,
    struct ggml_context * ctx,
    struct ggml_cgraph * gf,
    struct ggml_tensor * src0,
    struct ggml_tensor * src1,
    bool reduce_results
) {
    if (!tp || !tp->initialized) {
        return ggml_mul_mat(ctx, src0, src1);
    }

    struct ggml_tensor * result = ggml_mul_mat(ctx, src0, src1);

    if (reduce_results && tp->tp_size > 1) {
        struct ggml_tensor * result_reduced = ggml_dup(ctx, result);
        ggml_tp_all_reduce_sum(tp, ctx, result, result_reduced);
        ggml_build_forward_expand(gf, result_reduced);
        return result_reduced;
    }

    return result;
}

struct ggml_tensor * ggml_cuda_tp_rope(
    struct ggml_tensor_parallel * tp,
    struct ggml_context * ctx,
    struct ggml_tensor * src0,
    struct ggml_tensor * src1,
    struct ggml_tensor * src2,
    int n_dims,
    int mode,
    float freq_base,
    float freq_scale,
    float ext_factor,
    float attn_factor,
    float beta_fast,
    float beta_slow
) {
    return ggml_rope_ext(
        ctx, src0, src1, src2, n_dims, mode, 0,
        freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow
    );
}

struct ggml_tensor * ggml_cuda_tp_attention(
    struct ggml_tensor_parallel * tp,
    struct ggml_context * ctx,
    struct ggml_tensor * query,
    struct ggml_tensor * key,
    struct ggml_tensor * value,
    struct ggml_tensor * mask,
    float scale
) {
    return ggml_flash_attn_ext(
        ctx, query, key, value, mask,
        scale, 0.0f, 0.0f
    );
}

struct ggml_cuda_tp_layer * ggml_cuda_tp_layer_init(
    struct ggml_tensor_parallel * tp,
    int n_heads,
    int n_kv_heads,
    int hidden_size,
    int intermediate_size
) {
    struct ggml_cuda_tp_layer * layer = (struct ggml_cuda_tp_layer *) calloc(1, sizeof(struct ggml_cuda_tp_layer));
    if (!layer) return NULL;

    layer->tp = tp;
    layer->n_heads = n_heads;
    layer->n_kv_heads = n_kv_heads;
    layer->hidden_size = hidden_size;
    layer->intermediate_size = intermediate_size;

    return layer;
}

void ggml_cuda_tp_layer_free(struct ggml_cuda_tp_layer * layer) {
    if (!layer) return;
    free(layer);
}

struct ggml_tensor * ggml_cuda_tp_layer_forward(
    struct ggml_cuda_tp_layer * layer,
    struct ggml_context * ctx,
    struct ggml_cgraph * gf,
    struct ggml_tensor * hidden_states,
    struct ggml_tensor * position_embeddings,
    struct ggml_tensor * kv_self
) {
    struct ggml_tensor_parallel * tp = layer->tp;

    int tp_rank = ggml_tp_get_rank(tp);
    int tp_size = ggml_tp_get_size(tp);

    int head_dim = layer->hidden_size / layer->n_heads;
    int n_kv_heads_actual = layer->n_kv_heads / tp_size;
    float scale = 1.0f / sqrtf((float) head_dim);

    struct ggml_tensor * query = ggml_mul_mat(ctx, layer->q_proj, hidden_states);
    struct ggml_tensor * key = ggml_mul_mat(ctx, layer->k_proj, hidden_states);
    struct ggml_tensor * value = ggml_mul_mat(ctx, layer->v_proj, hidden_states);

    if (tp_size > 1) {
        struct ggml_tensor * query_reduced = ggml_dup(ctx, query);
        struct ggml_tensor * key_reduced = ggml_dup(ctx, key);
        struct ggml_tensor * value_reduced = ggml_dup(ctx, value);

        ggml_tp_all_reduce_sum(tp, ctx, query, query_reduced);
        ggml_tp_all_reduce_sum(tp, ctx, key, key_reduced);
        ggml_tp_all_reduce_sum(tp, ctx, value, value_reduced);

        query = query_reduced;
        key = key_reduced;
        value = value_reduced;
    }

    struct ggml_tensor * query_rope = ggml_cuda_tp_rope(
        tp, ctx, query, position_embeddings, NULL,
        head_dim, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f
    );

    struct ggml_tensor * key_rope = ggml_cuda_tp_rope(
        tp, ctx, key, position_embeddings, NULL,
        head_dim, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f
    );

    struct ggml_tensor * attn_output = ggml_cuda_tp_attention(
        tp, ctx, query_rope, key_rope, value, NULL, scale
    );

    struct ggml_tensor * output = ggml_mul_mat(ctx, layer->o_proj, attn_output);

    if (tp_size > 1) {
        struct ggml_tensor * output_reduced = ggml_dup(ctx, output);
        ggml_tp_all_reduce_sum(tp, ctx, output, output_reduced);
        output = output_reduced;
    }

    return output;
}
