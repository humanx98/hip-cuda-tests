cmake_minimum_required(VERSION 3.22.1)
project(hip-mem-transfers LANGUAGES C CXX VERSION 0.0.1)

set(CMAKE_CPP_STANDARD 23)
set(CMAKE_CPP_STANDARD_REQUIRED ON)
set(CMAKE_COMPILE_WARNING_AS_ERROR ON)

include(FetchContent)
set(FETCHCONTENT_QUIET FALSE)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(src/stb_impl)

find_file(CUDA_EXECUTABLE hipcc)
if(CUDA_EXECUTABLE)
    add_subdirectory(src/cuda)
else()
    message(WARNING "Cuda not found.")
endif()

find_file(HIPCC_EXECUTABLE hipcc)
if(HIPCC_EXECUTABLE)
    add_subdirectory(src/hip)
else()
    message(WARNING "Hip not found.")
endif()
