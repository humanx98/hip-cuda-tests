#include "hip_common.h"
#include <array>
#include <hip/hip_runtime.h>
#include <iostream>
#include <filesystem>
#include <ostream>
#include <vector>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image.h>
#include <stb_image_write.h>
#include <string>
#include <chrono>

#define LOOP_ITERATIONS 60000
#define ENABLE_TRANSFER_OFFSETS

int random(int min, int max){
   return min + std::rand() / (RAND_MAX / (max - min + 1) + 1);
}

struct Memory {
    std::vector<uint8_t> host_mem;
    hipDeviceptr_t device_mem;
};

void mem_transfers(int argc, const char* argv[]) {
    std::cout << "HIP Mem transfers" << std::endl;
    int device_index = 0;
    std::filesystem::path kernel_path = std::filesystem::path(argv[0]).parent_path() / "hip_kernels.hipfb";

    hip_print_devices(device_index);
    HIP_CHECK(hipInit(0));
    HIP_CHECK(hipSetDevice(device_index));
    hipStream_t stream;
    HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    hipModule_t module;
    HIP_CHECK(hipModuleLoad(&module, kernel_path.c_str()));
    hipFunction_t kernel_func;
    HIP_CHECK(hipModuleGetFunction(&kernel_func, module, "kernel_main"));

    int2 res = {};
    res.x = 3840;
    res.y = 2160;
    int num_components = 4;
    std::array<Memory, 3> mems;
    for (Memory& m : mems) {
        m.host_mem.resize(res.x * res.y * num_components);
        hipDeviceptr_t device_pixels;
        HIP_CHECK(hipMalloc(&m.device_mem, m.host_mem.size()));
    }

    constexpr size_t min_transfer = 24;
    constexpr size_t max_transfer = 1024;
    std::vector<size_t> random_offsets;
    std::vector<size_t> random_transfer_sizes;
    random_offsets.reserve(LOOP_ITERATIONS);    
    random_transfer_sizes.reserve(LOOP_ITERATIONS);
    for (size_t i = 0; i < LOOP_ITERATIONS; i++) {
    #ifdef ENABLE_TRANSFER_OFFSETS
        random_offsets.push_back(random(0, mems[0].host_mem.size() - max_transfer));
    #else
        random_offsets.push_back(0);
    #endif
        random_transfer_sizes.push_back(random(min_transfer, max_transfer));
    }

    int2 threads;
    threads.x = 32;
    threads.y = 32;
    int3 blocks;
    blocks.x = (res.x + threads.x - 1) / threads.x;
    blocks.y = (res.y + threads.y - 1) / threads.y;
    blocks.z = 1;
    std::cout << "Start launching kernel + DtoH transfers" << std::endl;
    auto begin = std::chrono::steady_clock::now();
    for (size_t i = 0; i < LOOP_ITERATIONS; i++) {
        Memory& m = mems[i % 3];
        void* args[] = { &m.device_mem, &res };

        HIP_CHECK(hipModuleLaunchKernel(kernel_func, blocks.x, blocks.y, blocks.z, threads.x, threads.y, 1, 0, stream, args, 0));
        HIP_CHECK(hipMemcpyDtoHAsync(m.host_mem.data(), ((uint8_t*)m.device_mem) + random_offsets[i], random_transfer_sizes[i], stream));
        HIP_CHECK(hipStreamSynchronize(stream));

        // HIP_CHECK(hipMemcpyDtoHAsync(m.host_mem.data(), m.device_mem, m.host_mem.size(), stream));
        // HIP_CHECK(hipStreamSynchronize(stream));
        // std::string png_name = std::to_string(i) + "_image.png";
        // stbi_write_png(png_name.c_str(), res.x, res.y, 4, m.host_mem.data(), res.x * num_components);
    }
    auto end = std::chrono::steady_clock::now();
    std::cout << "finished, time = "  << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() / 1000.0  << "s"<< std::endl;
   
    
    for (Memory& m : mems) {
        HIP_CHECK(hipFree(m.device_mem));
    }
    HIP_CHECK(hipModuleUnload(module));
    HIP_CHECK(hipStreamDestroy(stream));
}
