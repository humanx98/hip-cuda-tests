#include "common.h"
#include <cstdlib>
#include <cstdio>

void cuda_check(const char * file, const int line, cudaError_t err, const char* expression) {
    if (err != cudaSuccess) {
        printf("%s:%d: \"%s\" returned error %d\n", file, line, expression, err);
        std::abort();
    }
}

void cu_check(const char * file, const int line, CUresult err, const char* expression) {
    if (err != CUDA_SUCCESS) {
        printf("%s:%d: \"%s\" returned error %d\n", file, line, expression, err);
        std::abort();
    }
}

void cuda_print_devices(int selected_device) {
    int device_count = -1;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    printf("devices:\n");
    for (int i = 0; i < device_count; i++) {
        cudaDeviceProp properties;
        CUDA_CHECK(cudaGetDeviceProperties(&properties, i));
        if (selected_device == i) {
            printf("\t%d. %s (selected)\n", i, properties.name);
        } else {
            printf("\t%d. %s\n", i, properties.name);
        }
    }

    if (selected_device >= device_count) {
        printf("seleceted device =%d is out fo range\n", selected_device);
        std::abort();
    }
}