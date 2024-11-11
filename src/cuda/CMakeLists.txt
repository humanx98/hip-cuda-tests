find_package(CUDAToolkit REQUIRED)

set(cuda_kernels_compile_options
    -arch=sm_86
    --cubin ${CMAKE_CURRENT_SOURCE_DIR}/kernels.cu
    -m 64
    -o ${CMAKE_CURRENT_BINARY_DIR}/cuda_kernels.cubin
    -Wno-deprecated-gpu-targets
    -ccbin="/usr/bin/cc"
    -std=c++17)    

add_custom_target(
    cuda_kernels
    COMMAND ${CUDAToolkit_NVCC_EXECUTABLE}  ${cuda_kernels_compile_options}
    COMMENT "Compiling cuda kernels"
)

set(HEADERS
    cases/pass_writes.h
    cases/mem_transfers.h
    common.h
)

set(SOURCES
    cases/pass_writes.cpp
    cases/mem_transfers.cpp
    common.cpp
    main.cpp
)

add_executable(cuda_app ${HEADERS} ${SOURCES})
add_dependencies(cuda_app cuda_kernels)
target_link_libraries(cuda_app CUDA::cudart_static CUDA::cuda_driver stb_impl)
target_include_directories(cuda_app PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
