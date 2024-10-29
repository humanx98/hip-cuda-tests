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
set(hip_binaries_arch gfx1030 gfx1031 gfx90c CACHE STRING "AMD HIP architectures to build binaries for")
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
    -D HIPCC
    -ffast-math
    -o ${CMAKE_CURRENT_BINARY_DIR}/hip_kernels.hipfb)

add_custom_target(
    hip_kernels
    COMMAND ${HIPCC_EXECUTABLE}  ${hip_kernels_compile_options}
    COMMENT "Compiling hip kernels"
)

set(HEADERS
    cases/mem_transfers.h
    hip_common.h
)

set(SOURCES
    cases/mem_transfers.cpp
    hip_common.cpp
    main.cpp
)

add_executable(app ${HEADERS} ${SOURCES})
add_dependencies(app hip_kernels)
target_link_libraries(app hip::host stb_impl)
target_include_directories(app PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
