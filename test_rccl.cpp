#include <rccl.h>
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>

#define RCCL_CHECK(call)                                                       \
    do {                                                                       \
        ncclResult_t err = call;                                                \
        if (err != ncclSuccess) {                                              \
            fprintf(stderr, "RCCL error at %s:%d: %s\n",                        \
                    __FILE__, __LINE__, ncclGetErrorString(err));               \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

#define HIP_CHECK(call)                                                        \
    do {                                                                       \
        hipError_t err = call;                                                  \
        if (err != hipSuccess) {                                               \
            fprintf(stderr, "HIP error at %s:%d: %s\n",                         \
                    __FILE__, __LINE__, hipGetErrorString(err));                \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

void run_all_reduce_test(int nGPUs, const int* devices, size_t N) {
    fprintf(stderr, "\n=== RCCL AllReduce Test ===\n");
    fprintf(stderr, "GPUs: %d, Elements: %zu\n\n", nGPUs, N);

    std::vector<ncclComm_t> comms(nGPUs);
    std::vector<hipStream_t> streams(nGPUs);
    std::vector<void*> sendbuffs(nGPUs);
    std::vector<void*> recvbuffs(nGPUs);
    std::vector<void*> host_buffs(nGPUs);

    ncclUniqueId uniqueId;
    RCCL_CHECK(ncclGetUniqueId(&uniqueId));

    for (int i = 0; i < nGPUs; i++) {
        HIP_CHECK(hipSetDevice(devices[i]));
        HIP_CHECK(hipStreamCreate(&streams[i]));
        HIP_CHECK(hipMalloc(&sendbuffs[i], N * sizeof(float)));
        HIP_CHECK(hipMalloc(&recvbuffs[i], N * sizeof(float)));
        host_buffs[i] = malloc(N * sizeof(float));

        for (size_t j = 0; j < N; j++) {
            ((float*)host_buffs[i])[j] = (float)(i + 1);
        }
        HIP_CHECK(hipMemcpy(sendbuffs[i], host_buffs[i], N * sizeof(float), hipMemcpyHostToDevice));
    }

    for (int i = 0; i < nGPUs; i++) {
        HIP_CHECK(hipSetDevice(devices[i]));
        RCCL_CHECK(ncclCommInitRank(&comms[i], nGPUs, uniqueId, i));
        fprintf(stderr, "Rank %d initialized on GPU %d\n", i, devices[i]);
    }

    fprintf(stderr, "\nRunning AllReduce...\n");

    auto start = std::chrono::high_resolution_clock::now();

    RCCL_CHECK(ncclGroupStart());
    for (int i = 0; i < nGPUs; i++) {
        HIP_CHECK(hipSetDevice(devices[i]));
        RCCL_CHECK(ncclAllReduce(sendbuffs[i], recvbuffs[i], N,
                                  ncclFloat32, ncclSum, comms[i], streams[i]));
    }
    RCCL_CHECK(ncclGroupEnd());

    for (int i = 0; i < nGPUs; i++) {
        HIP_CHECK(hipSetDevice(devices[i]));
        HIP_CHECK(hipStreamSynchronize(streams[i]));
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    fprintf(stderr, "AllReduce completed in %.3f ms\n", duration.count() / 1000.0f);

    float expected_sum = 0.0f;
    for (int i = 0; i < nGPUs; i++) {
        expected_sum += (float)(i + 1);
    }

    bool success = true;
    for (int i = 0; i < nGPUs; i++) {
        HIP_CHECK(hipSetDevice(devices[i]));
        HIP_CHECK(hipMemcpy(host_buffs[i], recvbuffs[i], N * sizeof(float), hipMemcpyDeviceToHost));

        float* results = (float*)host_buffs[i];
        for (size_t j = 0; j < std::min(N, (size_t)10); j++) {
            if (fabs(results[j] - expected_sum) > 1e-5) {
                fprintf(stderr, "ERROR at GPU %d, element %zu: expected %.1f, got %.1f\n",
                        i, j, expected_sum, results[j]);
                success = false;
            }
        }
    }

    if (success) {
        fprintf(stderr, "\nAllReduce test PASSED!\n");
        fprintf(stderr, "First 10 elements: ");
        for (size_t j = 0; j < 10; j++) {
            fprintf(stderr, "%.1f ", ((float*)host_buffs[0])[j]);
        }
        fprintf(stderr, "\n");
    } else {
        fprintf(stderr, "\nAllReduce test FAILED!\n");
    }

    for (int i = 0; i < nGPUs; i++) {
        HIP_CHECK(hipSetDevice(devices[i]));
        HIP_CHECK(hipFree(sendbuffs[i]));
        HIP_CHECK(hipFree(recvbuffs[i]));
        HIP_CHECK(hipStreamDestroy(streams[i]));
        free(host_buffs[i]);
        ncclCommDestroy(comms[i]);
    }
}

int main(int argc, char** argv) {
    int nGPUs;
    HIP_CHECK(hipGetDeviceCount(&nGPUs));

    fprintf(stderr, "=== RCCL Multi-GPU Communication Test ===\n");
    fprintf(stderr, "Found %d HIP devices\n\n", nGPUs);

    for (int i = 0; i < nGPUs; i++) {
        hipDeviceProp_t prop;
        HIP_CHECK(hipGetDeviceProperties(&prop, i));
        fprintf(stderr, "GPU %d: %s (%.1f GB)\n", i, prop.name, prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    }

    if (nGPUs < 2) {
        fprintf(stderr, "\nThis test requires at least 2 GPUs. Found %d.\n", nGPUs);
        return 1;
    }

    int devices[2] = {0, 1};
    size_t N = 1024 * 1024;

    if (argc > 1) {
        N = atoll(argv[1]);
    }

    run_all_reduce_test(2, devices, N);

    fprintf(stderr, "\n=== Test Complete ===\n");

    return 0;
}
