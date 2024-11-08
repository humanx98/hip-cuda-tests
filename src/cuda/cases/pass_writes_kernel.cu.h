#pragma once

#include <cstdint>
#include <hip/hip_runtime.h>

#define PASS_COUNT 5

#define PASS_ANY (~0)
#define PASS0_WRITE (1 << 0)
#define PASS1_WRITE (1 << 1)
#define PASS2_WRITE (1 << 2)
#define PASS3_WRITE (1 << 3)

#define atomic_add_and_fetch_float(p, x) (atomicAdd((float *)(p), (float)(x)) + (float)(x))

struct KernelParams {
    size_t pass_offsets[PASS_COUNT];
    float iteration;
    int pass_writes;
};

__constant__ KernelParams kernel_params;

__device__ float4 film_write_pass_float4(float *__restrict__ buffer, float4 value)
{
    float *buf_x = buffer + 0;
    float *buf_y = buffer + 1;
    float *buf_z = buffer + 2;
    float *buf_w = buffer + 3;

    return {
        atomic_add_and_fetch_float(buf_x, value.x),
        atomic_add_and_fetch_float(buf_y, value.y),
        atomic_add_and_fetch_float(buf_z, value.z),
        atomic_add_and_fetch_float(buf_w, value.w),
    };
}

extern "C" __global__ void pass_writes_kernel(float* pixels, int2 resolution)
{
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t x = index % resolution.x;
    const uint32_t y = index / resolution.x;

    if (x < resolution.x && y < resolution.y) {
        const uint32_t index = x + y * resolution.x;
        float4 value = {
            x / (float)resolution.x,
            y / (float)resolution.y,
            0.0f,
            1.0f,
        };

        float4 acc_value = { 0.0f , 0.0f, 0.0f, 0.0f };
        int pass_writes = kernel_params.pass_writes;
        if (pass_writes & PASS_ANY) {

            if (pass_writes & PASS0_WRITE) {
                const int offset = 0;
                film_write_pass_float4(pixels + kernel_params.pass_offsets[offset] + index * 4, value);
            }

            if (pass_writes & PASS1_WRITE) {
                const int offset = 1;
                acc_value = film_write_pass_float4(pixels + kernel_params.pass_offsets[offset] + index * 4, value);
            }

            if (pass_writes & PASS2_WRITE) {
                const int offset = 2;
                film_write_pass_float4(pixels + kernel_params.pass_offsets[offset] + index * 4, value);
            }

            if (pass_writes & PASS3_WRITE) {
                const int offset = 3;
                film_write_pass_float4(pixels + kernel_params.pass_offsets[offset] + index * 4, value);
            }
        }
        pixels[kernel_params.pass_offsets[4] + index * 4 + 0] = acc_value.x / kernel_params.iteration;
        pixels[kernel_params.pass_offsets[4] + index * 4 + 1] = acc_value.y / kernel_params.iteration;
        pixels[kernel_params.pass_offsets[4] + index * 4 + 2] = acc_value.z / kernel_params.iteration;
        pixels[kernel_params.pass_offsets[4] + index * 4 + 3] = acc_value.w / kernel_params.iteration;
    }
}