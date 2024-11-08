#include "common.h"
#include <cstdlib>
#include <cstdio>

void hip_check(const char * file, const int line, hipError_t err, const char* expression) {
    if (err != hipSuccess) {
        printf("%s:%d: \"%s\" returned error %d\n", file, line, expression, err);
        std::abort();
    }
}

void hip_print_devices(int selected_device) {
    int device_count = -1;
    HIP_CHECK(hipGetDeviceCount(&device_count));

    printf("devices:\n");
    for (int i = 0; i < device_count; i++) {
        hipDeviceProp_t properties;
        HIP_CHECK(hipGetDeviceProperties(&properties, i));
        if (selected_device == i) {
            printf("\t%d. %s (%s, selected)\n", i, properties.name, properties.gcnArchName);
        } else {
            printf("\t%d. %s (%s)\n", i, properties.name, properties.gcnArchName);
        }
    }

    if (selected_device >= device_count) {
        printf("seleceted device =%d is out fo range\n", selected_device);
        std::abort();
    }
}