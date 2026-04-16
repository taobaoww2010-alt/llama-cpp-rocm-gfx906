#include "ggml-tensor-parallel.h"
#include "ggml-tensor-parallel/ggml-tp-comm.h"
#include "ggml/ggml.h"
#include "ggml/ggml-alloc.h"
#include "ggml/ggml-backend-impl.h"
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
    tp->device_ids = calloc(n_devices, sizeof(int));
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
    if (!tp || !tp->initialized || tp->tp_size <= 1) {
        if (tensor != dst) {
            ggml_build_forward_expand(ggml_cpy(ctx, tensor, dst));
        }
        return;
    }

    int device_id = tp->device_ids[tp->tp_rank];

    struct ggml_tensor * result = ggml_add_inplace(ctx, dst, tensor);
    if (!result) {
        result = ggml_add(ctx, dst, tensor);
    }

    ggml_build_forward_expand(result);
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
    if (!tp || !tp->initialized || tp->tp_size <= 1) {
        if (tensor != dst) {
            ggml_build_forward_expand(ggml_cpy(ctx, tensor, dst));
        }
        return;
    }

    struct ggml_tensor * result = ggml_cont(ctx, ggml_permute(ctx, tensor, 0, 2, 1, 3));
    ggml_build_forward_expand(result);
}

void ggml_tp_reduce_scatter(
    struct ggml_tensor_parallel * tp,
    struct ggml_context * ctx,
    struct ggml_tensor * tensor,
    struct ggml_tensor * dst,
    int dim
) {
    if (!tp || !tp->initialized || tp->tp_size <= 1) {
        if (tensor != dst) {
            ggml_build_forward_expand(ggml_cpy(ctx, tensor, dst));
        }
        return;
    }

    ggml_tp_all_reduce_sum(tp, ctx, tensor, dst);
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
    return ggml_rope_ext(ctx, inp, pos, n_head, n_head_kv, 0, freq_base,
                         freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
}

struct ggml_tensor * ggml_tp_cpy(
    struct ggml_tensor_parallel * tp,
    struct ggml_context * ctx,
    struct ggml_tensor * src,
    struct ggml_tensor * dst
) {
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
    return ggml_permute(ctx, tensor, axis0, axis1, axis2, axis3);
}

struct ggml_tensor * ggml_tp_cont(
    struct ggml_tensor_parallel * tp,
    struct ggml_context * ctx,
    struct ggml_tensor * tensor
) {
    return ggml_cont(ctx, tensor);
}

struct ggml_tensor * ggml_tp_view_1d(
    struct ggml_tensor_parallel * tp,
    struct ggml_context * ctx,
    struct ggml_tensor * tensor,
    int ne1,
    size_t offset1
) {
    return ggml_view_1d(ctx, tensor, ne1, offset1);
}

struct ggml_tensor * ggml_tp_view_2d(
    struct ggml_tensor_parallel * tp,
    struct ggml_context * ctx,
    struct ggml_tensor * tensor,
    int ne1,
    int ne2,
    size_t nb1,
    size_t nb2,
    size_t offset
) {
    return ggml_view_2d(ctx, tensor, ne1, ne2, nb1, nb2, offset);
}

struct ggml_tensor * ggml_tp_reshape_3d(
    struct ggml_tensor_parallel * tp,
    struct ggml_context * ctx,
    struct ggml_tensor * tensor,
    int ne1,
    int ne2,
    int ne3
) {
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
    return ggml_reshape_4d(ctx, tensor, ne1, ne2, ne3, ne4);
}

void ggml_tp_build_backward(
    struct ggml_tensor_parallel * tp,
    struct ggml_cgraph * fg,
    struct ggml_cgraph * bg,
    bool keep
) {
    ggml_build_backward(ctx, fg, bg, keep);
}
