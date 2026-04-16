#include "llama-tensor-parallel.h"
#include "llama-model.h"
#include "llama-arch.h"
#include "ggml-tensor-parallel.h"
#include "ggml/ggml.h"
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>

llama_tensor_parallel::llama_tensor_parallel()
    : tp_rank_(0), tp_size_(1), initialized_(false), tp_ctx_(nullptr) {
}

llama_tensor_parallel::~llama_tensor_parallel() {
    free();
}

bool llama_tensor_parallel::init(int n_gpus) {
    if (n_gpus < 1 || n_gpus > GGML_TP_MAX_DEVICES) {
        fprintf(stderr, "llama_tensor_parallel: invalid n_gpus=%d\n", n_gpus);
        return false;
    }

    tp_size_ = n_gpus;
    tp_rank_ = 0;

    int * device_ids = (int *)malloc(n_gpus * sizeof(int));
    for (int i = 0; i < n_gpus; i++) {
        device_ids[i] = i;
    }

    ggml_tp_init(&tp_, n_gpus, device_ids);

    free(device_ids);

    initialized_ = tp_.initialized;
    return initialized_;
}

void llama_tensor_parallel::free() {
    if (initialized_) {
        ggml_tp_free(&tp_);
        initialized_ = false;
    }
}

struct ggml_tensor * llama_tensor_parallel::compute_qkv(
    struct ggml_context * ctx,
    struct ggml_tensor * hidden_states,
    struct llama_layer * layer,
    int layer_idx
) {
    struct ggml_tensor * cur = hidden_states;

    cur = ggml_norm(ctx, cur, layer->attn_norm, layer->attn_norm_b);

    struct ggml_tensor * q = ggml_mul_mat(ctx, layer->wq, cur);
    struct ggml_tensor * k = ggml_mul_mat(ctx, layer->wk, cur);
    struct ggml_tensor * v = ggml_mul_mat(ctx, layer->wv, cur);

    if (tp_size_ > 1) {
        struct ggml_tensor * q_reduced = ggml_dup(ctx, q);
        struct ggml_tensor * k_reduced = ggml_dup(ctx, k);
        struct ggml_tensor * v_reduced = ggml_dup(ctx, v);

        ggml_tp_all_reduce_sum(&tp_, ctx, q, q_reduced);
        ggml_tp_all_reduce_sum(&tp_, ctx, k, k_reduced);
        ggml_tp_all_reduce_sum(&tp_, ctx, v, v_reduced);

        q = q_reduced;
        k = k_reduced;
        v = v_reduced;
    }

    return ggml_concat(ctx, q, k, v, 2);
}

struct ggml_tensor * llama_tensor_parallel::compute_attention(
    struct ggml_context * ctx,
    struct ggml_tensor * qkv,
    struct ggml_tensor * pos_ids,
    struct llama_layer * layer,
    int n_heads,
    int n_kv_heads
) {
    struct ggml_tensor * q = ggml_view_1d(ctx, qkv, ggml_nelements(qkv) / 3, 0);
    struct ggml_tensor * k = ggml_view_1d(ctx, qkv, ggml_nelements(qkv) / 3, ggml_nelements(qkv) / 3);
    struct ggml_tensor * v = ggml_view_1d(ctx, qkv, ggml_nelements(qkv) / 3, 2 * ggml_nelements(qkv) / 3);

    int head_dim = ggml_nelements(q) / n_heads;
    q = ggml_reshape_4d(ctx, q, head_dim, n_heads, -1, -1);
    k = ggml_reshape_4d(ctx, k, head_dim, n_kv_heads, -1, -1);
    v = ggml_reshape_4d(ctx, v, head_dim, n_kv_heads, -1, -1);

    q = ggml_rope_ext(ctx, q, pos_ids, n_heads, n_kv_heads, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(ctx, k, pos_ids, n_heads, n_kv_heads, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    struct ggml_tensor * attn_output = ggml_flash_attn_ext(
        ctx, q, k, v, nullptr, false, n_heads, n_kv_heads, 0.0f, -1.0f, false
    );

    attn_output = ggml_reshape_2d(ctx, attn_output, -1, -1);

    struct ggml_tensor * output = ggml_mul_mat(ctx, layer->wo, attn_output);

    if (tp_size_ > 1) {
        struct ggml_tensor * output_reduced = ggml_dup(ctx, output);
        ggml_tp_all_reduce_sum(&tp_, ctx, output, output_reduced);
        output = output_reduced;
    }

    return output;
}

struct ggml_tensor * llama_tensor_parallel::compute_ffn(
    struct ggml_context * ctx,
    struct ggml_tensor * hidden_states,
    struct llama_layer * layer
) {
    struct ggml_tensor * gate = ggml_mul_mat(ctx, layer->ffn_gate, hidden_states);
    struct ggml_tensor * up = ggml_mul_mat(ctx, layer->ffn_up, hidden_states);

    if (tp_size_ > 1) {
        struct ggml_tensor * gate_reduced = ggml_dup(ctx, gate);
        struct ggml_tensor * up_reduced = ggml_dup(ctx, up);

        ggml_tp_all_reduce_sum(&tp_, ctx, gate, gate_reduced);
        ggml_tp_all_reduce_sum(&tp_, ctx, up, up_reduced);

        gate = gate_reduced;
        up = up_reduced;
    }

    gate = ggml_silu(ctx, gate);
    gate = ggml_mul(ctx, gate, up);

    struct ggml_tensor * down = ggml_mul_mat(ctx, layer->ffn_down, gate);

    if (tp_size_ > 1) {
        struct ggml_tensor * down_reduced = ggml_dup(ctx, down);
        ggml_tp_all_reduce_sum(&tp_, ctx, down, down_reduced);
        down = down_reduced;
    }

    return down;
}

void llama_tensor_parallel::barrier() {
    if (initialized_) {
        ggml_tp_barrier(&tp_);
    }
}

void llama_tensor_parallel::sync_all() {
    if (initialized_) {
        ggml_tp_sync(&tp_);
    }
}

std::unique_ptr<llama_tensor_parallel> ggml_llama_tp_init(
    int n_gpus,
    const int * device_ids
) {
    auto tp = std::make_unique<llama_tensor_parallel>();
    if (!tp->init(n_gpus)) {
        return nullptr;
    }
    return tp;
}

void ggml_llama_tp_build_graph(
    llama_tensor_parallel * tp,
    struct llama_model * model,
    struct ggml_context * ctx,
    struct ggml_cgraph * gf,
    const struct llama_sequences & sequences,
    const struct llama_token_data_array * candidates_prompt,
    const struct llama_token_data_array * candidates_gen,
    bool embeddings
) {
    fprintf(stderr, "llama_tensor_parallel: build_graph called\n");
}
