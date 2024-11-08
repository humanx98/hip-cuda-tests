cmake_minimum_required(VERSION 3.22.1)
project(hip-mem-transfers LANGUAGES C CXX VERSION 0.0.1)

set(CMAKE_CPP_STANDARD 23)
set(CMAKE_CPP_STANDARD_REQUIRED ON)
set(CMAKE_COMPILE_WARNING_AS_ERROR ON)

include(FetchContent)
set(FETCHCONTENT_QUIET FALSE)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(src/stb_impl)

find_package(CUDAToolkit)
if(CUDAToolkit_FOUND)
    add_subdirectory(src/cuda)
else()
    message(WARNING "Cuda not found.")
endif()

if(UNIX)
    if(NOT DEFINED ROCM_PATH)
        set(ROCM_PATH "/opt/rocm" CACHE STRING "Default ROCM installation directory.")
    endif()
    # Search for rocm in common locations
    list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH})
endif()

find_package(hip)
if(hip_FOUND)
    add_subdirectory(src/hip)
else()
    message(WARNING "Hip not found.")
endif()

if (NOT CUDAToolkit_FOUND AND NOT hip_FOUND)
    message(FATAL_ERROR "Cuda and hip not found")
endif()
