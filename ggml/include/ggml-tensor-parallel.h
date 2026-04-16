#pragma once

#include "ggml.h"
#include "ggml-tensor-parallel/ggml-tp-comm.h"

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_TP_MAX_DEVICES 8

struct ggml_tensor_parallel {
    int n_devices;
    int tp_size;
    int tp_rank;
    int * device_ids;

    struct ggml_tp_comm * comm;

    bool initialized;
};

void ggml_tp_init(
    struct ggml_tensor_parallel * tp,
    int n_devices,
    const int * device_ids
);

void ggml_tp_free(struct ggml_tensor_parallel * tp);

int ggml_tp_get_rank(const struct ggml_tensor_parallel * tp);

int ggml_tp_get_size(const struct ggml_tensor_parallel * tp);

int ggml_tp_get_local_rank(const struct ggml_tensor_parallel * tp);

void ggml_tp_all_reduce_sum(
    struct ggml_tensor_parallel * tp,
    struct ggml_context * ctx,
    struct ggml_tensor * tensor,
    struct ggml_tensor * dst
);

void ggml_tp_all_reduce_sum_inplace(
    struct ggml_tensor_parallel * tp,
    struct ggml_context * ctx,
    struct ggml_tensor * tensor
);

void ggml_tp_all_gather(
    struct ggml_tensor_parallel * tp,
    struct ggml_context * ctx,
    struct ggml_tensor * tensor,
    struct ggml_tensor * dst,
    int dim
);

void ggml_tp_reduce_scatter(
    struct ggml_tensor_parallel * tp,
    struct ggml_context * ctx,
    struct ggml_tensor * tensor,
    struct ggml_tensor * dst,
    int dim
);

void ggml_tp_barrier(struct ggml_tensor_parallel * tp);

void ggml_tp_sync(struct ggml_tensor_parallel * tp);

struct ggml_tensor * ggml_tp_rope(
    struct ggml_tensor_parallel * tp,
    struct ggml_context * ctx,
    struct ggml_tensor * inp,
    struct ggml_tensor * pos,
    int n_head,
    int n_head_kv,
    float freq_base,
    float freq_scale,
    float ext_factor,
    float attn_factor,
    float beta_fast,
    float beta_slow
);

struct ggml_tensor * ggml_tp_cpy(
    struct ggml_tensor_parallel * tp,
    struct ggml_context * ctx,
    struct ggml_tensor * src,
    struct ggml_tensor * dst
);

struct ggml_tensor * ggml_tp_permute(
    struct ggml_tensor_parallel * tp,
    struct ggml_context * ctx,
    struct ggml_tensor * tensor,
    int axis0,
    int axis1,
    int axis2,
    int axis3
);

struct ggml_tensor * ggml_tp_cont(
    struct ggml_tensor_parallel * tp,
    struct ggml_context * ctx,
    struct ggml_tensor * tensor
);

struct ggml_tensor * ggml_tp_view_1d(
    struct ggml_tensor_parallel * tp,
    struct ggml_context * ctx,
    struct ggml_tensor * tensor,
    int ne1,
    size_t offset1
);

struct ggml_tensor * ggml_tp_view_2d(
    struct ggml_tensor_parallel * tp,
    struct ggml_context * ctx,
    struct ggml_tensor * tensor,
    int ne1,
    int ne2,
    size_t nb1,
    size_t offset
);

struct ggml_tensor * ggml_tp_reshape_3d(
    struct ggml_tensor_parallel * tp,
    struct ggml_context * ctx,
    struct ggml_tensor * tensor,
    int ne1,
    int ne2,
    int ne3
);

struct ggml_tensor * ggml_tp_reshape_4d(
    struct ggml_tensor_parallel * tp,
    struct ggml_context * ctx,
    struct ggml_tensor * tensor,
    int ne1,
    int ne2,
    int ne3,
    int ne4
);

void ggml_tp_build_backward(
    struct ggml_tensor_parallel * tp,
    struct ggml_cgraph * fg,
    struct ggml_cgraph * bg,
    bool keep
);

#ifdef __cplusplus
}
#endif
