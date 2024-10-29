#include <hip/hip_runtime.h>

extern "C" __global__ void kernel_main(uint8_t* pixels, int2 res)
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < res.x && y < res.y) {
        const uint32_t index = x + y * res.x;
        pixels[index * 4 + 0] = (x / (float)res.x) * 255;
        pixels[index * 4 + 1] = (y / (float)res.y) * 255;
        pixels[index * 4 + 2] = 0;
        pixels[index * 4 + 3] = 255;
    }
}