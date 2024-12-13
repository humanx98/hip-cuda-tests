if(UNIX)
    if(NOT DEFINED ROCM_PATH)
        set(ROCM_PATH "/opt/rocm" CACHE STRING "Default ROCM installation directory.")
    endif()
    # Search for rocm in common locations
    list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH})
endif()

find_package(hip REQUIRED)

find_file(HIPCC_EXECUTABLE hipcc HINTS ${ROCM_PATH} PATH_SUFFIXES bin)
if(NOT HIPCC_EXECUTABLE)
    message(FATAL_ERROR "hipcc is missing")
endif()

# set(hip_binaries_arch 
#     gfx1100 gfx1101 gfx1102 gfx1103 gfx1150 gfx1151  # Navi3
#     gfx1030 gfx1031 gfx1032 gfx1033 gfx1034 gfx1035 gfx1036  # Navi2
#     gfx1010 gfx1011 gfx1012 gfx1013  # Navi1
#     gfx900 gfx902 gfx904 gfx906 gfx908 gfx909 gfx90a gfx90c gfx940 gfx941 gfx942
#     CACHE STRING "AMD HIP architectures to build binaries for")
set(hip_binaries_arch)
set(target_gpus)
foreach(arch ${hip_binaries_arch})
    list(APPEND target_gpus "--offload-arch=${arch}")
endforeach()

set(hip_kernels_compile_options
    -O3
    -std=c++17
    ${target_gpus}
    -fgpu-rdc
    --genco
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels.hip.cpp
    -I ${CMAKE_CURRENT_SOURCE_DIR}/
    -o ${CMAKE_CURRENT_BINARY_DIR}/hip_kernels.hipfb)

add_custom_target(
    hip_kernels
    COMMAND ${HIPCC_EXECUTABLE}  ${hip_kernels_compile_options}
    COMMENT "Compiling hip kernels"
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

add_executable(hip_app ${HEADERS} ${SOURCES})
add_dependencies(hip_app hip_kernels)
target_link_libraries(hip_app hip::host stb_impl)
target_include_directories(hip_app PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
