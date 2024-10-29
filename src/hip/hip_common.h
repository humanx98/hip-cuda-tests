#pragma once

#include <hip/hip_runtime.h>

void hip_check(const char * file, const int line, hipError_t err, const char* expression);
#define HIP_CHECK(err) hip_check(__FILE__, __LINE__, err, #err)
void hip_print_devices(int selected_device);