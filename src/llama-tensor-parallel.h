#pragma once

#include "llama.h"
#include "ggml-tensor-parallel.h"
#include <memory>
#include <vector>

struct llama_tp_layer {
    struct ggml_tensor * attn_q;
    struct ggml_tensor * attn_k;
    struct ggml_tensor * attn_v;
    struct ggml_tensor * attn_o;

    struct ggml_tensor * ffn_gate;
    struct ggml_tensor * ffn_up;
    struct ggml_tensor * ffn_down;
};

class llama_tensor_parallel {
public:
    llama_tensor_parallel();
    ~llama_tensor_parallel();

    bool init(int n_gpus);
    void free();

    int get_rank() const { return tp_rank_; }
    int get_size() const { return tp_size_; }
    bool is_initialized() const { return initialized_; }

    struct ggml_tensor_parallel * get_tp() { return &tp_; }

    struct ggml_tensor * compute_qkv(
        struct ggml_context * ctx,
        struct ggml_tensor * hidden_states,
        struct llama_layer * layer,
        int layer_idx
    );

    struct ggml_tensor * compute_attention(
        struct ggml_context * ctx,
        struct ggml_tensor * qkv,
        struct ggml_tensor * pos_ids,
        struct llama_layer * layer,
        int n_heads,
        int n_kv_heads
    );

    struct ggml_tensor * compute_ffn(
        struct ggml_context * ctx,
        struct ggml_tensor * hidden_states,
        struct llama_layer * layer
    );

    void barrier();
    void sync_all();

private:
    struct ggml_tensor_parallel tp_;
    int tp_rank_;
    int tp_size_;
    bool initialized_;

    struct ggml_context * tp_ctx_;
};

std::unique_ptr<llama_tensor_parallel> ggml_llama_tp_init(
    int n_gpus,
    const int * device_ids
);

void ggml_llama_tp_build_graph(
    llama_tensor_parallel * tp,
    struct llama_model * model,
    struct ggml_context * ctx,
    struct ggml_cgraph * gf,
    const struct llama_sequences & sequences,
    const struct llama_token_data_array * candidates_prompt,
    const struct llama_token_data_array * candidates_gen,
    bool embeddings
);
