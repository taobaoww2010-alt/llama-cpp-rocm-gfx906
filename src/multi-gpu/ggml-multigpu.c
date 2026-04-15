#include "llama.h"
#include "common.h"

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <map>
#include <atomic>
#include <mutex>
#include <thread>
#include <chrono>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <netinet/in.h>
#include <arpa/inet.h>

struct gpu_instance {
    int id;
    std::string name;
    size_t total_mem;
    size_t free_mem;
    std::atomic<int> active_requests;
    std::atomic<bool> healthy;
    pid_t pid;
    int port;

    gpu_instance(int i) : id(i), active_requests(0), healthy(true), pid(-1), port(0) {}
};

class gpu_scheduler {
public:
    gpu_scheduler();
    ~gpu_scheduler();

    bool init(int n_parallel);
    void release();

    int get_n_gpus() const { return (int)gpus_.size(); }
    const gpu_instance* get_gpu(int idx) const {
        return idx >= 0 && idx < (int)gpus_.size() ? &gpus_[idx] : nullptr;
    }

    int select_gpu();
    void release_gpu(int gpu_id);
    void start_instance(int gpu_id, const char* model_path, const char* model_alias);

    void update_stats();

private:
    std::vector<gpu_instance> gpus_;
    std::atomic<int> round_robin_;
    std::mutex mutex_;
    bool initialized_;
};

gpu_scheduler::gpu_scheduler() : round_robin_(0), initialized_(false) {}

gpu_scheduler::~gpu_scheduler() {
    release();
}

bool gpu_scheduler::init(int n_parallel) {
    if (initialized_) {
        release();
    }

    int n_devices = llama_max_devices();
    fprintf(stderr, "\n=== GPU Scheduler Initialization ===\n");
    fprintf(stderr, "Found %d CUDA/HIP devices\n", n_devices);

    gpus_.clear();

    for (int i = 0; i < n_devices; i++) {
        ggml_backend_dev_t dev = llama_get_device(i);
        if (!dev) continue;

        gpu_instance gpu(i);
        gpu.name = ggml_backend_dev_get_name(dev);

        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);

        if (props.memory) {
            ggml_backend_dev_memory(dev, &gpu.free_mem, &gpu.total_mem);
            fprintf(stderr, "GPU %d: %s\n", i, gpu.name.c_str());
            fprintf(stderr, "  Memory: %.1f GB total, %.1f GB free\n",
                    gpu.total_mem / (1024.0 * 1024.0 * 1024.0),
                    gpu.free_mem / (1024.0 * 1024.0 * 1024.0));
        }

        gpus_.push_back(gpu);
    }

    if (gpus_.empty()) {
        fprintf(stderr, "ERROR: No CUDA/HIP devices found!\n");
        return false;
    }

    initialized_ = true;
    fprintf(stderr, "\n=== GPU Scheduler: %zu device(s) available ===\n\n", gpus_.size());
    return true;
}

void gpu_scheduler::release() {
    for (auto& gpu : gpus_) {
        if (gpu.pid > 0) {
            kill(gpu.pid, SIGTERM);
            gpu.pid = -1;
        }
    }
    gpus_.clear();
    initialized_ = false;
}

int gpu_scheduler::select_gpu() {
    if (gpus_.empty()) return -1;

    int best_idx = 0;
    int min_load = gpus_[0].active_requests.load();

    for (size_t i = 1; i < gpus_.size(); i++) {
        int load = gpus_[i].active_requests.load();
        if (load < min_load) {
            min_load = load;
            best_idx = (int)i;
        }
    }

    gpus_[best_idx].active_requests.fetch_add(1);
    return best_idx;
}

void gpu_scheduler::release_gpu(int gpu_id) {
    for (auto& gpu : gpus_) {
        if (gpu.id == gpu_id) {
            gpu.active_requests.fetch_sub(1);
            return;
        }
    }
}

void gpu_scheduler::start_instance(int gpu_id, const char* model_path, const char* model_alias) {
    if (gpu_id < 0 || gpu_id >= (int)gpus_.size()) return;

    gpu_instance& gpu = gpus_[gpu_id];
    int base_port = 8080;

    pid_t pid = fork();
    if (pid == 0) {
        char port_str[16];
        char gpu_id_str[8];
        char n_parallel_str[16];

        snprintf(port_str, sizeof(port_str), "%d", base_port + gpu_id);
        snprintf(gpu_id_str, sizeof(gpu_id_str), "%d", gpu_id);
        snprintf(n_parallel_str, sizeof(n_parallel_str), "%d", 4);

        std::vector<const char*> args = {
            "llama-server",
            "-m", model_path,
            "-mg", gpu_id_str,
            "-c", "4096",
            "-np", n_parallel_str,
            "--port", port_str,
        };

        if (model_alias) {
            args.push_back("-alias");
            args.push_back(model_alias);
        }
        args.push_back(nullptr);

        fprintf(stderr, "[GPU %d] Starting server on port %s\n", gpu_id, port_str);
        execvp(args[0], (char* const*)args.data());
        _exit(1);
    } else if (pid > 0) {
        gpu.pid = pid;
        gpu.port = base_port + gpu_id;
        fprintf(stderr, "[GPU %d] Server started with PID %d\n", gpu_id, pid);
    }
}

void gpu_scheduler::update_stats() {
    for (auto& gpu : gpus_) {
        ggml_backend_dev_t dev = llama_get_device(gpu.id);
        if (dev && dev->memory) {
            ggml_backend_dev_memory(dev, &gpu.free_mem, &gpu.total_mem);
        }
    }
}

extern "C" {

int ggml_multigpu_init(void) {
    return 0;
}

const char* ggml_get_gpu_name(int gpu_id) {
    static thread_local std::string name;
    ggml_backend_dev_t dev = llama_get_device(gpu_id);
    if (dev) {
        name = ggml_backend_dev_get_name(dev);
        return name.c_str();
    }
    return "unknown";
}

size_t ggml_get_gpu_memory(int gpu_id, size_t* total, size_t* free) {
    ggml_backend_dev_t dev = llama_get_device(gpu_id);
    if (dev && dev->memory) {
        ggml_backend_dev_memory(dev, free, total);
        return *free;
    }
    if (total) *total = 0;
    if (free) *free = 0;
    return 0;
}

int ggml_get_optimal_gpu(int prefer_busy) {
    int n_devices = llama_max_devices();
    if (n_devices <= 0) return 0;

    int best_gpu = 0;
    size_t max_free = 0;

    for (int i = 0; i < n_devices; i++) {
        ggml_backend_dev_t dev = llama_get_device(i);
        if (dev && dev->memory) {
            size_t total, free;
            ggml_backend_dev_memory(dev, &free, &total);
            if (free > max_free) {
                max_free = free;
                best_gpu = i;
            }
        }
    }

    return best_gpu;
}

}
