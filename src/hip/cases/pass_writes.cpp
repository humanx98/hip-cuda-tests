#include "pass_writes.h"
#include "hip_common.h"
#include "stb_impl.h"
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <iostream>

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
    hipDeviceptr_t device_mem;
};

void pass_writes(int device_index, const char* kernels_path) {
    HIP_CHECK(hipInit(0));
    HIP_CHECK(hipSetDevice(device_index));
    hipStream_t stream;
    HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    hipModule_t module;
    HIP_CHECK(hipModuleLoad(&module, kernels_path));
    hipFunction_t kernel_func;
    HIP_CHECK(hipModuleGetFunction(&kernel_func, module, "pass_writes_kernel"));

    int2 resolution = { 3840, 2160 };
    int num_components = 4;

    Pixels pixels;
    pixels.host_mem.resize(PASS_COUNT * resolution.x * resolution.y * num_components);
    HIP_CHECK(hipMalloc(&pixels.device_mem, vectorsizeof(pixels.host_mem)));
    HIP_CHECK(hipMemsetAsync(pixels.device_mem, 0, vectorsizeof(pixels.host_mem), stream));

    KernelParams host_kernel_params;
    host_kernel_params.iteration = 0.0f;
    host_kernel_params.pass_writes = PASS0_WRITE | PASS1_WRITE | PASS2_WRITE | PASS3_WRITE;
    for (size_t i = 0; i < PASS_COUNT; i++) {
        host_kernel_params.pass_offsets[i] = i * resolution.x * resolution.y * num_components;
    }

    size_t device_kernel_params_size;
    hipDeviceptr_t device_kernel_params;
    HIP_CHECK(hipModuleGetGlobal(&device_kernel_params, &device_kernel_params_size, module, "kernel_params"));
    if (sizeof(host_kernel_params) != device_kernel_params_size) {
        std::abort();
    }

    int3 threads = { 1024, 1, 1 };
    int3 blocks = { (resolution.x * resolution.y + threads.x - 1) / threads.x, 1, 1 };
    std::cout << "Start launching kernel + DtoH transfers" << std::endl;
    auto begin = std::chrono::steady_clock::now();
    for (size_t i = 0; i < LOOP_ITERATIONS; i++) {
        void* args[] = { &pixels.device_mem, &resolution };
        host_kernel_params.iteration += 1.0f;
        HIP_CHECK(hipMemcpyHtoDAsync(device_kernel_params, &host_kernel_params, device_kernel_params_size, stream));
        HIP_CHECK(hipModuleLaunchKernel(kernel_func, blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z, 0, stream, args, 0));
        HIP_CHECK(hipStreamSynchronize(stream));

        // HIP_CHECK(hipMemcpyDtoHAsync(pixels.host_mem.data(), pixels.device_mem, vectorsizeof(pixels.host_mem), stream));
        // HIP_CHECK(hipStreamSynchronize(stream));
        // stbi_write_hdr_as_png(std::to_string(i) + "_image.png", resolution.x, resolution.y, num_components, pixels.host_mem, host_kernel_params.pass_offsets[PASS_COUNT - 1]);
    }
    auto end = std::chrono::steady_clock::now();
    std::cout << "finished, time = "  << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() / 1000.0  << "s"<< std::endl;
   
    HIP_CHECK(hipFree(pixels.device_mem));
    HIP_CHECK(hipModuleUnload(module));
    HIP_CHECK(hipStreamDestroy(stream));
}