#pragma once

#include <hip/hip_runtime.h>

extern "C" __global__ void mem_transfers_kernel(float* pixels, int2 resolution)
{
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t x = index % resolution.x;
    const uint32_t y = index / resolution.x;

    if (x < resolution.x && y < resolution.y) {
        const uint32_t index = x + y * resolution.x;
        pixels[index * 4 + 0] = x / (float)resolution.x;
        pixels[index * 4 + 1] = y / (float)resolution.y;
        pixels[index * 4 + 2] = 0.0f;
        pixels[index * 4 + 3] = 1.0f;
    }
}