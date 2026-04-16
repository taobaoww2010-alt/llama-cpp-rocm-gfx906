#include "ggml-tensor-parallel.h"
#include "ggml-tensor-parallel/ggml-tp-comm.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend-impl.h"
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct ggml_tensor_parallel ggml_tp_default_instance = {0};

void ggml_tp_init(struct ggml_tensor_parallel * tp, int n_devices, const int * device_ids) {
    if (!tp) {
        tp = &ggml_tp_default_instance;
    }

    tp->n_devices = n_devices;
    tp->tp_size = n_devices;
    tp->tp_rank = 0;
    tp->device_ids = (int *)calloc(n_devices, sizeof(int));
    memcpy(tp->device_ids, device_ids, n_devices * sizeof(int));

    tp->comm = ggml_tp_comm_init(n_devices, device_ids);
    if (!tp->comm) {
        fprintf(stderr, "GGML TP: failed to initialize comm\n");
        return;
    }

    tp->tp_rank = ggml_tp_comm_get_rank(tp->comm);
    tp->initialized = true;

    fprintf(stderr, "GGML TP: initialized tensor parallel with %d devices, rank=%d\n",
            n_devices, tp->tp_rank);
}

void ggml_tp_free(struct ggml_tensor_parallel * tp) {
    if (!tp) {
        tp = &ggml_tp_default_instance;
    }

    if (tp->comm) {
        ggml_tp_comm_free(tp->comm);
        tp->comm = NULL;
    }

    free(tp->device_ids);
    tp->initialized = false;
}

int ggml_tp_get_rank(const struct ggml_tensor_parallel * tp) {
    if (!tp) {
        tp = &ggml_tp_default_instance;
    }
    return tp->tp_rank;
}

int ggml_tp_get_size(const struct ggml_tensor_parallel * tp) {
    if (!tp) {
        tp = &ggml_tp_default_instance;
    }
    return tp->tp_size;
}

int ggml_tp_get_local_rank(const struct ggml_tensor_parallel * tp) {
    if (!tp) {
        tp = &ggml_tp_default_instance;
    }
    return tp->tp_rank;
}

void ggml_tp_all_reduce_sum(
    struct ggml_tensor_parallel * tp,
    struct ggml_context * ctx,
    struct ggml_tensor * tensor,
    struct ggml_tensor * dst
) {
    (void)tp;
    (void)ctx;
    (void)tensor;
    (void)dst;
}

void ggml_tp_all_reduce_sum_inplace(
    struct ggml_tensor_parallel * tp,
    struct ggml_context * ctx,
    struct ggml_tensor * tensor
) {
    ggml_tp_all_reduce_sum(tp, ctx, tensor, tensor);
}

void ggml_tp_all_gather(
    struct ggml_tensor_parallel * tp,
    struct ggml_context * ctx,
    struct ggml_tensor * tensor,
    struct ggml_tensor * dst,
    int dim
) {
    (void)tp;
    (void)ctx;
    (void)tensor;
    (void)dst;
    (void)dim;
}

void ggml_tp_reduce_scatter(
    struct ggml_tensor_parallel * tp,
    struct ggml_context * ctx,
    struct ggml_tensor * tensor,
    struct ggml_tensor * dst,
    int dim
) {
    (void)tp;
    (void)ctx;
    (void)tensor;
    (void)dst;
    (void)dim;
}

void ggml_tp_barrier(struct ggml_tensor_parallel * tp) {
    if (!tp || !tp->initialized) return;
    ggml_tp_comm_barrier(tp->comm);
}

void ggml_tp_sync(struct ggml_tensor_parallel * tp) {
    if (!tp || !tp->initialized) return;
    ggml_tp_comm_sync_all(tp->comm);
}

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
) {
    (void)tp;
    (void)ctx;
    (void)inp;
    (void)pos;
    (void)n_head;
    (void)n_head_kv;
    (void)freq_base;
    (void)freq_scale;
    (void)ext_factor;
    (void)attn_factor;
    (void)beta_fast;
    (void)beta_slow;
    return NULL;
}

struct ggml_tensor * ggml_tp_cpy(
    struct ggml_tensor_parallel * tp,
    struct ggml_context * ctx,
    struct ggml_tensor * src,
    struct ggml_tensor * dst
) {
    (void)tp;
    return ggml_cpy(ctx, src, dst);
}

struct ggml_tensor * ggml_tp_permute(
    struct ggml_tensor_parallel * tp,
    struct ggml_context * ctx,
    struct ggml_tensor * tensor,
    int axis0,
    int axis1,
    int axis2,
    int axis3
) {
    (void)tp;
    return ggml_permute(ctx, tensor, axis0, axis1, axis2, axis3);
}

struct ggml_tensor * ggml_tp_cont(
    struct ggml_tensor_parallel * tp,
    struct ggml_context * ctx,
    struct ggml_tensor * tensor
) {
    (void)tp;
    return ggml_cont(ctx, tensor);
}

struct ggml_tensor * ggml_tp_view_1d(
    struct ggml_tensor_parallel * tp,
    struct ggml_context * ctx,
    struct ggml_tensor * tensor,
    int ne1,
    size_t offset1
) {
    (void)tp;
    return ggml_view_1d(ctx, tensor, ne1, offset1);
}

struct ggml_tensor * ggml_tp_view_2d(
    struct ggml_tensor_parallel * tp,
    struct ggml_context * ctx,
    struct ggml_tensor * tensor,
    int ne1,
    int ne2,
    size_t nb1,
    size_t offset
) {
    (void)tp;
    return ggml_view_2d(ctx, tensor, ne1, ne2, nb1, offset);
}

struct ggml_tensor * ggml_tp_reshape_3d(
    struct ggml_tensor_parallel * tp,
    struct ggml_context * ctx,
    struct ggml_tensor * tensor,
    int ne1,
    int ne2,
    int ne3
) {
    (void)tp;
    return ggml_reshape_3d(ctx, tensor, ne1, ne2, ne3);
}

struct ggml_tensor * ggml_tp_reshape_4d(
    struct ggml_tensor_parallel * tp,
    struct ggml_context * ctx,
    struct ggml_tensor * tensor,
    int ne1,
    int ne2,
    int ne3,
    int ne4
) {
    (void)tp;
    return ggml_reshape_4d(ctx, tensor, ne1, ne2, ne3, ne4);
}

void ggml_tp_build_backward(
    struct ggml_tensor_parallel * tp,
    struct ggml_cgraph * fg,
    struct ggml_cgraph * bg,
    bool keep
) {
    (void)tp;
    (void)fg;
    (void)bg;
    (void)keep;
}
