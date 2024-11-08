#include <iostream>
#include <filesystem>
#include "cases/pass_writes.h"
#include "cases/mem_transfers.h"
#include "common.h"

int main(int argc, const char* argv[]) {
    std::filesystem::path kernel_path = std::filesystem::path(argv[0]).parent_path() / "hip_kernels.hipfb";
    int case_num = 1;
    int device_index = 0;
    hip_print_devices(device_index);
    switch (case_num) {
        case 0: 
            std::cout << "case: mem transfers" << std::endl;
            mem_transfers(device_index, kernel_path.c_str());
            break;
        case 1: 
            std::cout << "case: pass writes" << std::endl;
            pass_writes(device_index, kernel_path.c_str());
            break;
    }
    return 0;
}
