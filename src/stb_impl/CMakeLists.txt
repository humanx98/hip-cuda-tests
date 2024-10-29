FetchContent_Declare(stb 
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
)
FetchContent_MakeAvailable(stb)

set(HEADERS stb_impl.h)
set(SOURCES stb_impl.cpp)

add_library(stb_impl STATIC ${SOURCES} ${HEADERS})
target_include_directories(stb_impl PUBLIC ${stb_SOURCE_DIR})

