cmake_minimum_required(VERSION 3.25)

project(rwkv-mobile CXX)

set(CMAKE_CXX_STANDARD 17)

include(FetchContent)

option(ENABLE_RWKVCPP_BACKEND "Enable RWKV.cpp backend" ON)
option(ENABLE_WEBRWKV_BACKEND "Enable WebRWKV backend" ON)

option(RWKV_MOBILE_BUILD_EXAMPLES "Build examples" ON)

set(RWKV_MOBILE_SRCS
    src/runtime.cpp
    src/sampler.cpp
    src/tokenizer.cpp
    src/logger.cpp
    src/c_api.cpp
)

if (ENABLE_WEBRWKV_BACKEND)
    set(RWKV_MOBILE_SRCS ${RWKV_MOBILE_SRCS} backends/web-rwkv/src/web_rwkv_backend.cpp)

    FetchContent_Declare(
        Corrosion
        GIT_REPOSITORY https://github.com/corrosion-rs/corrosion.git
        GIT_TAG v0.5 # Optionally specify a commit hash, version tag or branch here
    )
    FetchContent_MakeAvailable(Corrosion)

    corrosion_import_crate(MANIFEST_PATH backends/web-rwkv/Cargo.toml)
endif()

add_library(rwkv_mobile_internal ${RWKV_MOBILE_SRCS})
target_include_directories(rwkv_mobile_internal PUBLIC src)

if (ENABLE_WEBRWKV_BACKEND)
    if (APPLE)
        set(WEBRWKV_EXTRA_LIBS "-framework QuartzCore -framework Metal -lSystem -framework CoreGraphics -framework CoreFoundation -lobjc -liconv")
    endif()
    target_compile_definitions(rwkv_mobile_internal PUBLIC ENABLE_WEBRWKV)
    target_include_directories(rwkv_mobile_internal PUBLIC backends/web-rwkv)
    target_link_libraries(rwkv_mobile_internal PUBLIC web_rwkv_ffi ${WEBRWKV_EXTRA_LIBS})
endif()

if (RWKV_MOBILE_BUILD_EXAMPLES)
    add_executable(gen examples/gen.cpp)
    target_link_libraries(gen PUBLIC rwkv_mobile_internal)
endif()
