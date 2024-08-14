cmake_minimum_required(VERSION 3.25)

project(rwkv-mobile CXX)

set(CMAKE_CXX_STANDARD 17)

option(ENABLE_RWKVCPP_BACKEND "Enable RWKV.cpp backend" ON)

set(RWKV_MOBILE_SRCS
    src/model.cpp
    src/logger.cpp
)

if (DEFINED ANDROID_NDK)
    find_library(log-lib log)
    find_library(android-lib android)
    # target_link_libraries(faster_rwkv_internal PUBLIC ${log-lib} ${android-lib})
endif()