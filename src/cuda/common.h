#pragma once

#include <hip/hip_runtime.h>
#include <vector>

void hip_check(const char * file, const int line, hipError_t err, const char* expression);
#define HIP_CHECK(err) hip_check(__FILE__, __LINE__, err, #err)
void hip_print_devices(int selected_device);

template<typename T>
static inline size_t vectorsizeof(const typename std::vector<T>& vec) {
    return sizeof(T) * vec.size();
}