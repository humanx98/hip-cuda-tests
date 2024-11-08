#include "common.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <array>
#include <iostream>
#include <ostream>
#include <vector>
#include <stb_impl.h>
#include <chrono>

#define LOOP_ITERATIONS 60000
#define ENABLE_TRANSFER_OFFSETS

static int random(int min, int max){
   return min + std::rand() / (RAND_MAX / (max - min + 1) + 1);
}

struct Memory {
    std::vector<float> host_mem;
    void* device_mem;
};

void mem_transfers(int device_index, const char* kernels_path) {
    CU_CHECK(cuInit(0));
    CUDA_CHECK(cudaSetDevice(device_index));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUmodule module;
    CU_CHECK(cuModuleLoad(&module, kernels_path));
    cudaFunction_t kernel_func;
    CU_CHECK(cuModuleGetFunction(&kernel_func, module, "mem_transfers_kernel"));

    int2 resolution = { 3840, 2160 };
    int num_components = 4;
    std::array<Memory, 3> mems;
    for (Memory& m : mems) {
        m.host_mem.resize(resolution.x * resolution.y * num_components);
        void* device_pixels;
        CUDA_CHECK(cudaMalloc(&m.device_mem, vectorsizeof(m.host_mem)));
    }

    constexpr size_t min_transfer = 24;
    constexpr size_t max_transfer = 1024;
    std::vector<size_t> random_offsets;
    std::vector<size_t> random_transfer_sizes;
    random_offsets.reserve(LOOP_ITERATIONS);    
    random_transfer_sizes.reserve(LOOP_ITERATIONS);
    for (size_t i = 0; i < LOOP_ITERATIONS; i++) {
    #ifdef ENABLE_TRANSFER_OFFSETS
        random_offsets.push_back(random(0, vectorsizeof(mems[0].host_mem) - max_transfer));
    #else
        random_offsets.push_back(0);
    #endif
        random_transfer_sizes.push_back(random(min_transfer, max_transfer));
    }

    int3 threads = { 1024, 1, 1 };
    int3 blocks = { (resolution.x * resolution.y + threads.x - 1) / threads.x, 1, 1 };
    std::cout << "Start launching kernel + DtoH transfers" << std::endl;
    auto begin = std::chrono::steady_clock::now();
    for (size_t i = 0; i < LOOP_ITERATIONS; i++) {
        Memory& m = mems[i % 3];
        void* args[] = { &m.device_mem, &resolution };

        CU_CHECK(cuLaunchKernel(kernel_func, blocks.x, blocks.y, blocks.z, threads.x, threads.y, 1, 0, stream, args, 0));
        CU_CHECK(cuMemcpyDtoHAsync(m.host_mem.data(), (CUdeviceptr)(((uint8_t*)m.device_mem) + random_offsets[i]), random_transfer_sizes[i], stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // CU_CHECK(cuMemcpyDtoHAsync(m.host_mem.data(), (CUdeviceptr)m.device_mem, vectorsizeof(m.host_mem), stream));
        // CUDA_CHECK(cudaStreamSynchronize(stream));
        // stbi_write_hdr_as_png(std::to_string(i) + "_image.png", resolution.x, resolution.y, num_components, m.host_mem);
    }
    auto end = std::chrono::steady_clock::now();
    std::cout << "finished, time = "  << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() / 1000.0  << "s"<< std::endl;
   
    
    for (Memory& m : mems) {
        CUDA_CHECK(cudaFree(m.device_mem));
    }
    CU_CHECK(cuModuleUnload(module));
    CUDA_CHECK(cudaStreamDestroy(stream));
}
