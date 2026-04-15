#pragma once

#include "llama.h"

#include <vector>
#include <memory>
#include <atomic>
#include <mutex>

struct llama_model;
struct llama_context;

struct model_instance {
    int device_id;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model;
    std::unique_ptr<llama_context, decltype(&llama_context_free)> context;
    std::atomic<int> busy_slots;
    std::atomic<int64_t> last_used;

    model_instance(int device, llama_model* m, llama_context* c)
        : device_id(device)
        , model(m, llama_model_free)
        , context(c, llama_context_free)
        , busy_slots(0)
        , last_used(0) {}
};

class multi_gpu_scheduler {
public:
    multi_gpu_scheduler();
    ~multi_gpu_scheduler();

    bool init(const char* model_path, int n_gpu_layers, int n_ctx, int n_parallel, int n_threads);
    void release();

    int get_n_devices() const { return (int)instances_.size(); }
    model_instance* get_instance(int idx);

    int assign_slot();
    void release_slot(int instance_idx);

    void set_tensor_parallelism(bool enable) { tensor_parallelism_ = enable; }
    bool is_tensor_parallelism() const { return tensor_parallelism_; }

    struct ggml_context* get_graph_ctx(int instance_idx) const;

private:
    std::vector<std::unique_ptr<model_instance>> instances_;
    std::vector<int> device_ids_;
    std::atomic<int> round_robin_;
    std::atomic<bool> tensor_parallelism_;
    std::mutex mutex_;

    int get_available_instance();
};

multi_gpu_scheduler::multi_gpu_scheduler()
    : round_robin_(0)
    , tensor_parallelism_(false) {
}

multi_gpu_scheduler::~multi_gpu_scheduler() {
    release();
}

bool multi_gpu_scheduler::init(const char* model_path, int n_gpu_layers, int n_ctx, int n_parallel, int n_threads) {
    std::lock_guard<std::mutex> lock(mutex_);

    llama_model_params mparams = llama_model_params_default();
    mparams.n_gpu_layers = n_gpu_layers;

    llama_context_params cparams = llama_context_params_default();
    cparams.n_ctx = n_ctx;
    cparams.n_seq_max = n_parallel;
    cparams.n_threads = n_threads;
    cparams.n_threads_batch = n_threads;

    ggml_backend_dev_t* devices = nullptr;
    int n_devices = llama_max_devices();

    instances_.clear();
    device_ids_.clear();

    for (int i = 0; i < n_devices; i++) {
        ggml_backend_dev_t dev = llama_get_device(i);
        if (!dev) continue;

        const char* dev_name = ggml_backend_dev_get_name(dev);
        fprintf(stderr, "multi_gpu_scheduler: checking device %d: %s\n", i, dev_name);

        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);

        if (props.memory) {
            size_t total, free;
            ggml_backend_dev_memory(dev, &free, &total);
            fprintf(stderr, "  GPU %d: %.1f GB total, %.1f GB free\n",
                    i, total / (1024.0 * 1024.0 * 1024.0),
                    free / (1024.0 * 1024.0 * 1024.0));
        }
    }

    int instances_per_gpu = 1;
    for (int gpu = 0; gpu < n_devices && (int)instances_.size() < (int)llama_max_devices(); gpu++) {
        ggml_backend_dev_t dev = llama_get_device(gpu);
        if (!dev) continue;

        device_ids_.push_back(gpu);
        mparams.devices = &dev;
        mparams.main_gpu = gpu;

        llama_model* model = llama_model_load_from_file(model_path, mparams);
        if (!model) {
            fprintf(stderr, "multi_gpu_scheduler: failed to load model on GPU %d\n", gpu);
            continue;
        }

        llama_context* ctx = llama_init_from_model(model, cparams);
        if (!ctx) {
            fprintf(stderr, "multi_gpu_scheduler: failed to create context on GPU %d\n", gpu);
            llama_model_free(model);
            continue;
        }

        fprintf(stderr, "multi_gpu_scheduler: initialized model instance on GPU %d\n", gpu);
        instances_.emplace_back(std::make_unique<model_instance>(gpu, model, ctx));

        if (tensor_parallelism_ && instances_.size() >= 2) {
            break;
        }
    }

    if (instances_.empty()) {
        fprintf(stderr, "multi_gpu_scheduler: failed to initialize any model instance\n");
        return false;
    }

    fprintf(stderr, "multi_gpu_scheduler: initialized %zu model instance(s)\n", instances_.size());
    return true;
}

void multi_gpu_scheduler::release() {
    std::lock_guard<std::mutex> lock(mutex_);
    instances_.clear();
    device_ids_.clear();
}

model_instance* multi_gpu_scheduler::get_instance(int idx) {
    if (idx < 0 || idx >= (int)instances_.size()) {
        return nullptr;
    }
    return instances_[idx].get();
}

int multi_gpu_scheduler::assign_slot() {
    int idx = get_available_instance();
    if (idx >= 0 && idx < (int)instances_.size()) {
        instances_[idx]->busy_slots.fetch_add(1);
        instances_[idx]->last_used.store(ggml_time_ms());
    }
    return idx;
}

void multi_gpu_scheduler::release_slot(int instance_idx) {
    if (instance_idx >= 0 && instance_idx < (int)instances_.size()) {
        instances_[instance_idx]->busy_slots.fetch_sub(1);
    }
}

int multi_gpu_scheduler::get_available_instance() {
    if (instances_.size() == 1) {
        return 0;
    }

    if (tensor_parallelism_) {
        return 0;
    }

    int best_idx = 0;
    int min_busy = instances_[0]->busy_slots.load();

    for (size_t i = 1; i < instances_.size(); i++) {
        int busy = instances_[i]->busy_slots.load();
        if (busy < min_busy) {
            min_busy = busy;
            best_idx = (int)i;
        }
    }

    return best_idx;
}
