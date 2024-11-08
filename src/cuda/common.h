#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <vector>

void cuda_check(const char * file, const int line, cudaError_t err, const char* expression);
#define CUDA_CHECK(err) cuda_check(__FILE__, __LINE__, err, #err)

void cu_check(const char * file, const int line, CUresult err, const char* expression);
#define CU_CHECK(err) cu_check(__FILE__, __LINE__, err, #err)

void cuda_print_devices(int selected_device);

template<typename T>
static inline size_t vectorsizeof(const typename std::vector<T>& vec) {
    return sizeof(T) * vec.size();
}