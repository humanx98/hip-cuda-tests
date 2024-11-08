#include "pass_writes.h"
#include "common.h"
#include "stb_impl.h"
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <chrono>

#define LOOP_ITERATIONS 10000

#define PASS_COUNT 5
#define PASS0_WRITE 1 << 0
#define PASS1_WRITE 1 << 1
#define PASS2_WRITE 1 << 2
#define PASS3_WRITE 1 << 3

struct KernelParams {
    size_t pass_offsets[PASS_COUNT];
    float iteration;
    int pass_writes;
};

struct Pixels {
    std::vector<float> host_mem;
    void* device_mem;
};

void pass_writes(int device_index, const char* kernels_path) {
    CU_CHECK(cuInit(0));
    CUDA_CHECK(cudaSetDevice(device_index));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUmodule module;
    CU_CHECK(cuModuleLoad(&module, kernels_path));
    cudaFunction_t kernel_func;
    CU_CHECK(cuModuleGetFunction(&kernel_func, module, "pass_writes_kernel"));

    int2 resolution = { 3840, 2160 };
    int num_components = 4;

    Pixels pixels;
    pixels.host_mem.resize(PASS_COUNT * resolution.x * resolution.y * num_components);
    CUDA_CHECK(cudaMalloc(&pixels.device_mem, vectorsizeof(pixels.host_mem)));
    CUDA_CHECK(cudaMemsetAsync(pixels.device_mem, 0, vectorsizeof(pixels.host_mem), stream));

    KernelParams host_kernel_params;
    host_kernel_params.iteration = 0.0f;
    host_kernel_params.pass_writes = PASS0_WRITE | PASS1_WRITE | PASS2_WRITE | PASS3_WRITE;
    for (size_t i = 0; i < PASS_COUNT; i++) {
        host_kernel_params.pass_offsets[i] = i * resolution.x * resolution.y * num_components;
    }

    size_t device_kernel_params_size;
    CUdeviceptr device_kernel_params;
    CU_CHECK(cuModuleGetGlobal(&device_kernel_params, &device_kernel_params_size, module, "kernel_params"));
    if (sizeof(host_kernel_params) != device_kernel_params_size) {
        std::abort();
    }

    int3 threads = { 1024, 1, 1 };
    int3 blocks = { (resolution.x * resolution.y + threads.x - 1) / threads.x, 1, 1 };
    std::cout << "Start launching kernel" << std::endl;
    auto begin = std::chrono::steady_clock::now();
    for (size_t i = 0; i < LOOP_ITERATIONS; i++) {
        void* args[] = { &pixels.device_mem, &resolution };
        host_kernel_params.iteration += 1.0f;
        CU_CHECK(cuMemcpyHtoDAsync((CUdeviceptr)device_kernel_params, &host_kernel_params, device_kernel_params_size, stream));
        CU_CHECK(cuLaunchKernel(kernel_func, blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z, 0, stream, args, 0));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // CU_CHECK(cuMemcpyDtoHAsync(pixels.host_mem.data(), (CUdeviceptr)pixels.device_mem, vectorsizeof(pixels.host_mem), stream));
        // CUDA_CHECK(cudaStreamSynchronize(stream));
        // stbi_write_hdr_as_png(std::to_string(i) + "_image.png", resolution.x, resolution.y, num_components, pixels.host_mem, host_kernel_params.pass_offsets[PASS_COUNT - 1]);
    }
    auto end = std::chrono::steady_clock::now();
    std::cout << "finished, time = "  << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() / 1000.0  << "s"<< std::endl;
   
    CUDA_CHECK(cudaFree(pixels.device_mem));
    CU_CHECK(cuModuleUnload(module));
    CUDA_CHECK(cudaStreamDestroy(stream));
}