#pragma once

#include "ggml-cuda.h"
#include "ggml-tensor-parallel.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

void ggml_cuda_tp_init(
    struct ggml_tensor_parallel * tp,
    int n_devices,
    const int * device_ids
);

void ggml_cuda_tp_free(struct ggml_tensor_parallel * tp);

struct ggml_tensor * ggml_cuda_tp_mul_mat(
    struct ggml_tensor_parallel * tp,
    struct ggml_context * ctx,
    struct ggml_tensor * src0,
    struct ggml_tensor * src1,
    bool reduce_results
);

struct ggml_tensor * ggml_cuda_tp_rope(
    struct ggml_tensor_parallel * tp,
    struct ggml_context * ctx,
    struct ggml_tensor * src0,
    struct ggml_tensor * src1,
    struct ggml_tensor * src2,
    int n_head,
    int n_head_kv,
    float freq_base,
    float freq_scale,
    float ext_factor,
    float attn_factor,
    float beta_fast,
    float beta_slow,
    struct ggml_tensor * dst
);

struct ggml_tensor * ggml_cuda_tp_attention(
    struct ggml_tensor_parallel * tp,
    struct ggml_context * ctx,
    struct ggml_tensor * query,
    struct ggml_tensor * key,
    struct ggml_tensor * value,
    struct ggml_tensor * dst,
    int n_head,
    int n_head_kv
);

struct ggml_cuda_tp_layer {
    struct ggml_tensor * q_proj;
    struct ggml_tensor * k_proj;
    struct ggml_tensor * v_proj;
    struct ggml_tensor * o_proj;

    struct ggml_tensor * gate_proj;
    struct ggml_tensor * up_proj;
    struct ggml_tensor * down_proj;

    struct ggml_tensor_parallel * tp;
    int n_heads;
    int n_kv_heads;
    int hidden_size;
    int intermediate_size;
};

struct ggml_cuda_tp_layer * ggml_cuda_tp_layer_init(
    struct ggml_tensor_parallel * tp,
    int n_heads,
    int n_kv_heads,
    int hidden_size,
    int intermediate_size
);

void ggml_cuda_tp_layer_free(struct ggml_cuda_tp_layer * layer);

struct ggml_tensor * ggml_cuda_tp_layer_forward(
    struct ggml_cuda_tp_layer * layer,
    struct ggml_context * ctx,
    struct ggml_tensor * hidden_states,
    struct ggml_tensor * position_embeddings,
    struct ggml_tensor * kv_self
);

#ifdef __cplusplus
}
#endif
