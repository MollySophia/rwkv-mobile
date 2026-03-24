cmake_minimum_required(VERSION 3.25)

project(rwkv-mobile C CXX)

if(APPLE)
  include(CheckOBJCSourceCompiles)
  enable_language(OBJC)
endif()

set(CMAKE_CXX_STANDARD 17)

include(FetchContent)

option(ENABLE_NCNN_BACKEND "Enable NCNN backend" ON)
option(ENABLE_WEBRWKV_BACKEND "Enable WebRWKV backend" ON)
option(ENABLE_LLAMACPP_BACKEND "Enable llama.cpp backend" ON)
option(ENABLE_LLAMACPP_VULKAN "Enable Vulkan for llama.cpp backend" OFF)
option(ENABLE_QNN_BACKEND "Enable QNN backend" OFF)
option(ENABLE_MNN_BACKEND "Enable MNN backend" ON)
option(ENABLE_MTK_NP7_BACKEND "Enable MediaTek NP7 backend" OFF)
option(ENABLE_COREML_BACKEND "Enable CoreML backend" OFF)
option(ENABLE_MLX_BACKEND "Enable MLX backend" OFF)

option(ENABLE_VISION "Enable vision" OFF)
option(ENABLE_WHISPER "Enable whisper" OFF)
option(ENABLE_TTS "Enable TTS" OFF)

option(BUILD_EXAMPLES "Build examples" ON)
option(BUILD_STATIC_LIB "Build static library" OFF)
option(ENABLE_SERVER "Enable built-in HTTP server" ON)

if (ENABLE_TTS AND NOT ENABLE_MNN_BACKEND)
    message(FATAL_ERROR "TTS requires MNN backend")
endif()

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

option(ADDRESS_SANITIZER "Enable AddressSanitizer" OFF)

if (ANDROID)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g0")
endif()

if (ADDRESS_SANITIZER)
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
    # set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
endif()

set(RWKV_MOBILE_SRCS
    src/runtime.cpp
    src/sampler.cpp
    src/tokenizer.cpp
    src/logger.cpp
    src/soc_detect.cpp
    src/rmpack.cpp
    src/backend.cpp
)

if (ENABLE_SERVER)
    set(RWKV_MOBILE_SRCS ${RWKV_MOBILE_SRCS} src/rwkv_http_server.cpp)
endif()

if (WIN32)
    set(CMAKE_LINK_LIBRARY_USING_WHOLE_ARCHIVE "/WHOLEARCHIVE:<LIBRARY>")
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    cmake_host_system_information(RESULT CPU_NUMBER_OF_LOGICAL_CORES QUERY NUMBER_OF_LOGICAL_CORES)
    add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/MP${CPU_NUMBER_OF_LOGICAL_CORES}>")
else()
    set(CMAKE_LINK_LIBRARY_USING_WHOLE_ARCHIVE "LINKER:--whole-archive" "<LINK_ITEM>" "LINKER:--no-whole-archive")
endif()
set(CMAKE_LINK_LIBRARY_USING_WHOLE_ARCHIVE_SUPPORTED TRUE)

set(BUILD_SHARED_LIBS_DEFAULT OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (ENABLE_WEBRWKV_BACKEND)
    set(RWKV_MOBILE_SRCS ${RWKV_MOBILE_SRCS} src/backends/web_rwkv/web_rwkv_backend.cpp)

    # if (IOS)
    #     set(Rust_CARGO_TARGET "aarch64-apple-ios")
    # endif()

    if (NOT IOS)
        FetchContent_Declare(
            Corrosion
            GIT_REPOSITORY https://github.com/corrosion-rs/corrosion.git
            GIT_TAG v0.6.1
        )
        FetchContent_MakeAvailable(Corrosion)

        FetchContent_Declare(
            web_rwkv_ffi
            GIT_REPOSITORY https://github.com/MollySophia/web-rwkv-ffi
            GIT_TAG 2150ed0212079ab9c06f1621eb7563d933c58ce2
        )
        FetchContent_MakeAvailable(web_rwkv_ffi)

        corrosion_import_crate(MANIFEST_PATH ${web_rwkv_ffi_SOURCE_DIR}/Cargo.toml
            CRATE_TYPES staticlib)
    endif()

endif()

if (ENABLE_NCNN_BACKEND)
    set(RWKV_MOBILE_SRCS ${RWKV_MOBILE_SRCS} src/backends/ncnn/ncnn_rwkv_backend.cpp)

    if (WIN32)
        FetchContent_Declare(
            ncnn_prebuilt
            URL https://github.com/Tencent/ncnn/releases/download/20250503/ncnn-20250503-windows-vs2022.zip
        )
        FetchContent_MakeAvailable(ncnn_prebuilt)
    else()
        FetchContent_Declare(
            ncnn
            GIT_REPOSITORY https://github.com/Tencent/ncnn
            GIT_TAG master
        )
        # include(disable_unused_ncnn_layers)
        option(NCNN_BUILD_TOOLS "" OFF)
        option(NCNN_BUILD_EXAMPLES "" OFF)
        option(NCNN_BUILD_TESTS "" OFF)
        option(NCNN_BUILD_BENCHMARK "" OFF)
        option(NCNN_DISABLE_EXCEPTION "" OFF)
        FetchContent_MakeAvailable(ncnn)
    endif()
endif()

if (ENABLE_MNN_BACKEND)
    set(RWKV_MOBILE_SRCS ${RWKV_MOBILE_SRCS} src/backends/mnn/mnn_rwkv_backend.cpp)

    FetchContent_Declare(
        mnn
        GIT_REPOSITORY https://github.com/alibaba/MNN
        GIT_TAG 3.3.0
    )

    if (ANDROID OR (WIN32 AND (CMAKE_SYSTEM_PROCESSOR MATCHES "arm64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")))
        option(MNN_ARM82 "" ON)
    else()
        # TODO: fix iOS MNN_ARM82
        option(MNN_ARM82 "" OFF)
    endif()

    if (ANDROID)
        option(MNN_SUPPORT_BF16 "" ON)
        option(MNN_OPENCL "" ON)
        option(MNN_VULKAN "" ON)
    endif()

    if (APPLE)
        option(MNN_METAL "" ON)
        option(MNN_COREML "" ON)
    endif()

    option(MNN_LOW_MEMORY "" ON)
    option(MNN_SUPPORT_TRANSFORMER_FUSE "" ON)
    option(MNN_BUILD_TOOLS "" OFF)
    option(MNN_BUILD_SHARED_LIBS "" OFF)
    option(MNN_SEP_BUILD "" OFF)
    option(MNN_USE_SYSTEM_LIB "" ON)
    option(MNN_BUILD_TOOLS "" OFF)
    if (WIN32)
        if (CMAKE_SYSTEM_PROCESSOR MATCHES "arm64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
            option(MNN_KLEIDIAI "" OFF)
        endif()
    endif()
    FetchContent_MakeAvailable(mnn)
    target_compile_definitions(MNN PUBLIC _DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)
    # target_compile_definitions(Express PUBLIC _DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)
    target_compile_definitions(MNNCore PUBLIC _DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)
    target_compile_definitions(MNNCPU PUBLIC _DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)
endif()

if (ENABLE_MTK_NP7_BACKEND)
    set(RWKV_MOBILE_SRCS ${RWKV_MOBILE_SRCS} src/backends/mtk_np7/mtk_np7_backend.cpp)
endif()

if (ENABLE_LLAMACPP_BACKEND)
    set(RWKV_MOBILE_SRCS ${RWKV_MOBILE_SRCS}
        src/backends/llama_cpp/llama_cpp_backend.cpp
        src/embedding/rwkv_embedding.cpp
    )
    if (ENABLE_VISION)
        set(RWKV_MOBILE_SRCS ${RWKV_MOBILE_SRCS}
            src/multimodal/vision/vision_encoder.cpp
            src/multimodal/vision/preprocessor.cpp
        )
    endif()

    if (ENABLE_WHISPER)
        set(RWKV_MOBILE_SRCS ${RWKV_MOBILE_SRCS}
            src/multimodal/whisper/whisper.cpp
            src/multimodal/whisper/whisper_encoder.cpp
        )
    endif()

    FetchContent_Declare(
        llama_cpp
        GIT_REPOSITORY https://github.com/ggml-org/llama.cpp
        GIT_TAG 29771a0a4c16e82a3e32974fddcb3e7b85359f53
    )

    option(LLAMA_STANDALONE "" OFF)
    option(LLAMA_BUILD_TESTS "" OFF)
    option(GGML_LLAMAFILE "" OFF)
    if (BUILD_STATIC_LIB)
        option(BUILD_SHARED_LIBS "" OFF)
        option(GGML_BACKEND_DL "" OFF)
    endif()

    if (APPLE)
        option(GGML_METAL "" ON)
        option(GGML_NATIVE "" OFF)
        option(GGML_CPU_ARM_ARCH "" "armv8.6-a")
    endif()
    if ((WIN32 OR CMAKE_SYSTEM_NAME STREQUAL "Linux") AND CMAKE_SIZEOF_VOID_P EQUAL 8 AND ENABLE_LLAMACPP_VULKAN)
        option(GGML_VULKAN "" ON)
    endif()

    FetchContent_MakeAvailable(llama_cpp)
    target_compile_definitions(llama PUBLIC _DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)
    target_compile_definitions(ggml PUBLIC _DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)
    target_compile_definitions(ggml-base PUBLIC _DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)
    target_compile_definitions(ggml-cpu PUBLIC _DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)
endif()

if (ENABLE_QNN_BACKEND)
    if (WIN32)
        file(GLOB QNN_SRCS src/backends/qnn/rwkv-qualcomm/WrapperUtils/*.cpp src/backends/qnn/rwkv-qualcomm/PAL/src/common/*.cpp
                        src/backends/qnn/rwkv-qualcomm/PAL/src/windows/*.cpp)
        set(RWKV_MOBILE_SRCS ${RWKV_MOBILE_SRCS} src/backends/qnn/rwkv-qualcomm/Utils/DataUtil.cpp
                src/backends/qnn/rwkv-qualcomm/Utils/DynamicLoadUtil.cpp
                src/backends/qnn/rwkv-qualcomm/Utils/IOTensor.cpp
                src/backends/qnn/rwkv-qualcomm/Utils/Utils.cpp
                src/backends/qnn/rwkv-qualcomm/Utils/ClientBuffer.cpp
                src/backends/qnn/rwkv-qualcomm/Utils/dlwrap.cpp
                src/backends/qnn/rwkv-qualcomm/Utils/RpcMem.cpp)
    else()
        file(GLOB QNN_SRCS src/backends/qnn/rwkv-qualcomm/Utils/*.cpp src/backends/qnn/rwkv-qualcomm/WrapperUtils/*.cpp src/backends/qnn/rwkv-qualcomm/PAL/src/common/*.cpp
                        src/backends/qnn/rwkv-qualcomm/PAL/src/linux/*.cpp)
    endif()
    set(RWKV_MOBILE_SRCS ${RWKV_MOBILE_SRCS} ${QNN_SRCS})
    set(RWKV_MOBILE_SRCS ${RWKV_MOBILE_SRCS} src/backends/qnn/qnn_backend.cpp)
endif()

if (ENABLE_WHISPER OR ENABLE_TTS)
    set(RWKV_MOBILE_SRCS ${RWKV_MOBILE_SRCS}
        src/multimodal/utils/audio.cpp
    )

    if (ENABLE_TTS)
        set(RWKV_MOBILE_SRCS ${RWKV_MOBILE_SRCS}
            src/multimodal/tts/sparktts.cpp
            src/multimodal/tts/frontend_utils.cpp
            src/multimodal/tts/tts_properties.cpp
        )

        if (NOT WIN32)
            include(kaldifst)
        endif()
    endif()

    message(CHECK_START "Fetching Eigen3")
    FetchContent_Declare(
        Eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.3
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE
    )
    # note: To disable eigen tests,
    # you should put this code in a add_subdirectory to avoid to change
    # BUILD_TESTING for your own project too since variables are directory
    # scoped
    set(BUILD_TESTING OFF)
    set(EIGEN_BUILD_TESTING OFF)
    set(EIGEN_MPL2_ONLY ON)
    set(EIGEN_BUILD_PKGCONFIG OFF)
    set(EIGEN_BUILD_DOC OFF)
    FetchContent_MakeAvailable(Eigen)

    # FetchContent_Declare(
    #     kaldi_native_fbank
    #     GIT_REPOSITORY https://github.com/csukuangfj/kaldi-native-fbank/
    #     GIT_TAG v1.21.1
    # )
    # option(KALDI_NATIVE_FBANK_BUILD_PYTHON "" OFF)
    # option(KALDI_NATIVE_FBANK_BUILD_TESTS "" OFF)
    # set(BUILD_SHARED_LIBS_BACKUP ${BUILD_SHARED_LIBS})
    # set(BUILD_SHARED_LIBS OFF)
    # FetchContent_MakeAvailable(kaldi_native_fbank)
    # set(BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS_BACKUP})
endif()

if (ENABLE_COREML_BACKEND)
    set(RWKV_MOBILE_SRCS ${RWKV_MOBILE_SRCS} src/backends/coreml/coreml_rwkv_backend.cpp)
endif()

if (ENABLE_MLX_BACKEND)
    set(RWKV_MOBILE_SRCS ${RWKV_MOBILE_SRCS} src/backends/mlx/mlx_rwkv_backend.cpp)
endif()

if (WIN32)
    add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
    add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
endif()

add_library(rwkv_mobile_internal STATIC ${RWKV_MOBILE_SRCS})
target_include_directories(rwkv_mobile_internal PUBLIC src)
if (CMAKE_ANDROID_ARCH_ABI STREQUAL "arm64-v8a")
    target_link_options(rwkv_mobile_internal PRIVATE
        "-Wl,-z,max-page-size=16384"
        "-Wl,-z,common-page-size=16384"
    )
endif()

if (WIN32)
    target_compile_definitions(rwkv_mobile_internal PUBLIC _DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)
endif()

if (ENABLE_WHISPER OR ENABLE_TTS)
    target_compile_definitions(rwkv_mobile_internal PUBLIC ENABLE_TTS)
    target_link_libraries(rwkv_mobile_internal PUBLIC Eigen3::Eigen)
    if (ENABLE_TTS)
        if (NOT WIN32)
            target_link_libraries(rwkv_mobile_internal PUBLIC
                # $<TARGET_OBJECTS:kaldi-native-fbank-core>
                $<TARGET_OBJECTS:kaldifst_core>
                $<TARGET_OBJECTS:fst>
            )

            target_include_directories(rwkv_mobile_internal PUBLIC
                # ${kaldi_native_fbank_SOURCE_DIR}
                ${openfst_SOURCE_DIR}/src/include
                ${kaldifst_SOURCE_DIR}
            )
        # else()
            # target_link_libraries(rwkv_mobile_internal PUBLIC
            #     $<TARGET_OBJECTS:kaldi-native-fbank-core>
            # )
            # target_include_directories(rwkv_mobile_internal PUBLIC
            #     ${kaldi_native_fbank_SOURCE_DIR}
            # )
        endif()
    endif()
    target_include_directories(rwkv_mobile_internal PUBLIC ${Eigen_SOURCE_DIR}
        src/multimodal/tts src/multimodal/utils)
endif()

if (ENABLE_WEBRWKV_BACKEND)
    FetchContent_Declare(
        bzip2
        GIT_REPOSITORY https://gitlab.com/bzip2/bzip2
        GIT_TAG master
    )
    option(ENABLE_STATIC_LIB "" ON)
    option(ENABLE_STATIC_LIB_IS_PIC "" ON)
    option(ENABLE_APP "" OFF)
    FetchContent_MakeAvailable(bzip2)
    if (APPLE)
        find_library(FOUNDATION_FRAMEWORK Foundation)
        find_library(QUARTZCORE_FRAMEWORK QuartzCore)
        find_library(METAL_FRAMEWORK Metal)
        find_library(COREGRAPHICS_FRAMEWORK CoreGraphics)
        find_library(COREFOUNDATION_FRAMEWORK CoreFoundation)
        find_library(OBJC_LIBRARY objc)
        find_library(ICONV_LIBRARY iconv)
        target_link_libraries(rwkv_mobile_internal PUBLIC ${FOUNDATION_FRAMEWORK} ${QUARTZCORE_FRAMEWORK} ${METAL_FRAMEWORK} ${COREGRAPHICS_FRAMEWORK} ${COREFOUNDATION_FRAMEWORK} ${OBJC_LIBRARY} ${ICONV_LIBRARY})
    elseif (WIN32)
        set(WEBRWKV_EXTRA_LIBS ws2_32 opengl32 d3d12 d3dcompiler userenv kernel32 user32 ntdll bcrypt propsys RuntimeObject)
    endif()
    target_compile_definitions(rwkv_mobile_internal PUBLIC ENABLE_WEBRWKV)
    target_include_directories(rwkv_mobile_internal PUBLIC src/backends/web_rwkv ${web_rwkv_ffi_SOURCE_DIR}/include)
    set(WEBRWKV_FFI_LIB web_rwkv_ffi)
    if (IOS)
        # target_compile_options(rwkv_mobile_internal PUBLIC "-I${web_rwkv_ffi_SOURCE_DIR}/include")
        target_include_directories(rwkv_mobile_internal PUBLIC src/backends/web_rwkv/include)
        target_compile_options(rwkv_mobile_internal PUBLIC "-Isrc/backends/web_rwkv/include")
        set(WEBRWKV_FFI_LIB "${CMAKE_CURRENT_SOURCE_DIR}/src/backends/web_rwkv/prebuilt/aarch64-apple-ios/libweb_rwkv_ffi.a")
    endif()
    target_link_libraries(rwkv_mobile_internal PUBLIC ${WEBRWKV_FFI_LIB} ${WEBRWKV_EXTRA_LIBS} bz2_static)
endif()

if (ENABLE_NCNN_BACKEND)
    target_compile_definitions(rwkv_mobile_internal PUBLIC ENABLE_NCNN)
    target_include_directories(rwkv_mobile_internal PUBLIC src/backends/ncnn)
    if (WIN32)
        if (CMAKE_SYSTEM_PROCESSOR MATCHES "arm64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
            set(NCNN_ARCH "arm64")
        elseif(CMAKE_SIZEOF_VOID_P EQUAL 8)
            set(NCNN_ARCH "x64")
        else()
            set(NCNN_ARCH "x86")
        endif()
        message(STATUS "Using NCNN architecture: ${CMAKE_SYSTEM_PROCESSOR} (${NCNN_ARCH})")
        set(ncnn_DIR ${ncnn_prebuilt_SOURCE_DIR}/${NCNN_ARCH}/lib/cmake/ncnn)
        find_package(ncnn REQUIRED)
    endif()
    target_link_libraries(rwkv_mobile_internal PUBLIC "$<LINK_LIBRARY:WHOLE_ARCHIVE,ncnn>")
endif()

if (ENABLE_COREML_BACKEND)
    find_library(FOUNDATION_FRAMEWORK Foundation)
    find_library(COREML_FRAMEWORK CoreML)

    if (COREML_FRAMEWORK)
        message(STATUS "CoreML framework found")
    endif()

    add_library(rwkv_coreml STATIC
        src/backends/coreml/rwkv-coreml.h
        src/backends/coreml/rwkv-coreml.mm
        src/backends/coreml/rwkv_coreml_firstchunk_impl.h
        src/backends/coreml/rwkv_coreml_firstchunk_impl.m
        src/backends/coreml/rwkv_coreml_impl.h
        src/backends/coreml/rwkv_coreml_impl.m
        src/backends/coreml/rwkv_coreml_singlechunk_impl.h
        src/backends/coreml/rwkv_coreml_singlechunk_impl.m
    )

    target_include_directories(rwkv_coreml PUBLIC src)
    target_link_libraries(rwkv_coreml PUBLIC ${FOUNDATION_FRAMEWORK} ${COREML_FRAMEWORK})
    set_target_properties(rwkv_coreml PROPERTIES
        COMPILE_FLAGS "-fobjc-arc"
        XCODE_ATTRIBUTE_CLANG_ENABLE_OBJC_ARC YES
    )
    set_target_properties(rwkv_coreml PROPERTIES FOLDER "libs")

    target_link_libraries(rwkv_mobile_internal PUBLIC rwkv_coreml)

    target_compile_definitions(rwkv_mobile_internal PUBLIC ENABLE_COREML)
    target_include_directories(rwkv_mobile_internal PUBLIC src/backends/coreml)

    # add_executable(test_coreml src/backends/coreml/test_coreml.cpp)
    # target_link_libraries(test_coreml PUBLIC rwkv_coreml)
endif()

if (ENABLE_MLX_BACKEND)
    find_library(FOUNDATION_LIBRARY Foundation REQUIRED)
    find_library(METAL_FRAMEWORK    Metal      REQUIRED)
    find_library(Accelerate_LIBRARY Accelerate REQUIRED)
    target_compile_definitions(rwkv_mobile_internal PUBLIC ENABLE_MLX)
    target_include_directories(rwkv_mobile_internal PUBLIC src/backends/mlx src/backends/mlx/include)
    # Set MLX library path based on platform
    if (IOS)
        set(MLX_LIB_PATH ${CMAKE_CURRENT_SOURCE_DIR}/src/backends/mlx/prebuilt/ios-arm64/libMLXModelFFI.a)
        target_link_directories(rwkv_mobile_internal PUBLIC src/backends/mlx/prebuilt/ios-arm64)
    else()
        set(MLX_LIB_PATH ${CMAKE_CURRENT_SOURCE_DIR}/src/backends/mlx/prebuilt/macos-arm64/libMLXModelFFI.a)
        target_link_directories(rwkv_mobile_internal PUBLIC src/backends/mlx/prebuilt/macos-arm64)
    endif()

    # Link Swift runtime libraries for static library containing Swift code
    if (APPLE)
        # Find Swift toolchain path
        execute_process(
            COMMAND xcrun --find swift
            OUTPUT_VARIABLE SWIFT_PATH
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )

        if (SWIFT_PATH)
            # Extract toolchain usr directory
            # SWIFT_PATH = /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/swift
            # -> /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin
            get_filename_component(SWIFT_BIN_DIR "${SWIFT_PATH}" DIRECTORY)
            # -> /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr
            get_filename_component(SWIFT_USR_DIR "${SWIFT_BIN_DIR}" DIRECTORY)

            # Determine platform-specific Swift runtime path
            if (IOS)
                set(SWIFT_RUNTIME_SUBDIR "iphoneos")
            else()
                set(SWIFT_RUNTIME_SUBDIR "macosx")
            endif()

            # Swift runtime libraries are in toolchain/usr/lib/swift/{platform}
            # e.g., /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib/swift/macosx
            set(SWIFT_RUNTIME_PATH "${SWIFT_USR_DIR}/lib/swift/${SWIFT_RUNTIME_SUBDIR}")

            # Check if Swift runtime path exists
            if (EXISTS "${SWIFT_RUNTIME_PATH}/libswiftCompatibility56.a")
                # Add Swift runtime path to linker
                target_link_directories(rwkv_mobile_internal PUBLIC ${SWIFT_RUNTIME_PATH})

                # Link Swift compatibility libraries (required for static Swift libraries)
                target_link_libraries(rwkv_mobile_internal PUBLIC
                    "${SWIFT_RUNTIME_PATH}/libswiftCompatibility56.a"
                    "${SWIFT_RUNTIME_PATH}/libswiftCompatibilityConcurrency.a"
                    "${SWIFT_RUNTIME_PATH}/libswiftCompatibilityPacks.a"
                )

                message(STATUS "Swift runtime libraries found at ${SWIFT_RUNTIME_PATH}")
            else()
                message(WARNING "Swift runtime libraries not found at ${SWIFT_RUNTIME_PATH}")
            endif()
        else()
            message(WARNING "Swift toolchain not found. MLX backend may not link correctly.")
        endif()
    endif()

    # Link MLX static library with full path to ensure it's found
    target_link_libraries(rwkv_mobile_internal PUBLIC ${MLX_LIB_PATH} ${FOUNDATION_LIBRARY} ${METAL_FRAMEWORK} ${Accelerate_LIBRARY})
endif()

if (ENABLE_LLAMACPP_BACKEND)
    if (APPLE AND GGML_ACCELERATE)
        if (ACCELERATE_FRAMEWORK)
            set(LLAMACPP_EXTRA_LIBS ${LLAMACPP_EXTRA_LIBS} ${ACCELERATE_FRAMEWORK})
        endif()
    endif()

    if (GGML_METAL)
        find_library(FOUNDATION_LIBRARY Foundation REQUIRED)
        find_library(METAL_FRAMEWORK    Metal      REQUIRED)
        find_library(METALKIT_FRAMEWORK MetalKit   REQUIRED)
        add_compile_definitions(GGML_USE_METAL)

        set(LLAMACPP_EXTRA_LIBS ${LLAMACPP_EXTRA_LIBS}
            ${FOUNDATION_LIBRARY}
            ${METAL_FRAMEWORK}
            ${METALKIT_FRAMEWORK}
        )

        set(LLAMACPP_EXTRA_LIBS ${LLAMACPP_EXTRA_LIBS} $<TARGET_OBJECTS:ggml-metal> $<TARGET_OBJECTS:ggml-blas>)
    endif()

    if (GGML_VULKAN AND TARGET ggml-vulkan)
        add_compile_definitions(GGML_USE_VULKAN)
        set(LLAMACPP_EXTRA_LIBS ${LLAMACPP_EXTRA_LIBS} $<TARGET_OBJECTS:ggml-vulkan>)
    endif()

    if (ENABLE_VISION)
        target_include_directories(rwkv_mobile_internal PUBLIC src/multimodal/vision)
        target_compile_definitions(rwkv_mobile_internal PUBLIC ENABLE_VISION)
    endif()

    if (ENABLE_WHISPER)
        target_include_directories(rwkv_mobile_internal PUBLIC src/multimodal/whisper src/multimodal/utils)
        target_compile_definitions(rwkv_mobile_internal PUBLIC ENABLE_WHISPER)
    endif()

    target_compile_definitions(rwkv_mobile_internal PUBLIC ENABLE_LLAMACPP)
    target_include_directories(rwkv_mobile_internal PUBLIC src/backends/llama_cpp ${llama_cpp_SOURCE_DIR}/include ${llama_cpp_SOURCE_DIR}/src ${llama_cpp_SOURCE_DIR}/ggml/include)
    target_link_libraries(rwkv_mobile_internal PUBLIC $<TARGET_OBJECTS:llama> $<TARGET_OBJECTS:ggml> $<TARGET_OBJECTS:ggml-base> $<TARGET_OBJECTS:ggml-cpu> ${LLAMACPP_EXTRA_LIBS})
    if (GGML_VULKAN)
        if (TARGET Vulkan::Vulkan)
            target_link_libraries(rwkv_mobile_internal PUBLIC Vulkan::Vulkan)
        elseif (Vulkan_LIBRARY)
            target_link_libraries(rwkv_mobile_internal PUBLIC ${Vulkan_LIBRARY})
        endif()
    endif()
    find_package(OpenMP)
    if (OpenMP_FOUND)
        if (ANDROID)
            target_link_libraries(rwkv_mobile_internal PUBLIC -static-openmp)
        elseif(NOT IOS)
            target_link_libraries(rwkv_mobile_internal PUBLIC OpenMP::OpenMP_CXX)
        endif()
    endif()
endif()

if (ENABLE_QNN_BACKEND)
    target_compile_definitions(rwkv_mobile_internal PUBLIC ENABLE_QNN)
    target_include_directories(rwkv_mobile_internal PUBLIC src/backends/qnn
                               src/backends/qnn/rwkv-qualcomm
                               src/backends/qnn/rwkv-qualcomm/Utils
                               src/backends/qnn/rwkv-qualcomm/PAL/include
                               src/backends/qnn/rwkv-qualcomm/WrapperUtils
                               src/backends/qnn/rwkv-qualcomm/QNN/)
    if (WIN32)
        target_link_libraries(rwkv_mobile_internal PRIVATE Shlwapi Shell32)
    endif()
endif()

if (ENABLE_MNN_BACKEND)
    target_compile_definitions(rwkv_mobile_internal PUBLIC ENABLE_MNN)
    target_link_libraries(rwkv_mobile_internal PUBLIC
        $<LINK_LIBRARY:WHOLE_ARCHIVE,MNN>
    )
    target_include_directories(rwkv_mobile_internal PUBLIC src/backends/mnn ${mnn_SOURCE_DIR}/include)
endif()

if (ENABLE_MTK_NP7_BACKEND)
    if (NOT ANDROID)
        message(FATAL_ERROR "mtk_np7 backend is only supported on Android builds")
    endif()

    target_compile_definitions(rwkv_mobile_internal PUBLIC ENABLE_MTK_NP7)
    target_include_directories(rwkv_mobile_internal PUBLIC
        src/backends/mtk_np7
        src/backends/mtk_np7/include
    )

    add_library(rwkv_mtk_prebuilt STATIC IMPORTED)
    set_target_properties(rwkv_mtk_prebuilt PROPERTIES
        IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/src/backends/mtk_np7/prebuilt/${ANDROID_ABI}/librwkv_mtk.a"
    )
    target_link_libraries(rwkv_mobile_internal PUBLIC rwkv_mtk_prebuilt)
endif()

if (BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if (ENABLE_SERVER)
    FetchContent_Declare(
        cpp_httplib
        GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
        GIT_TAG master
    )
    option(HTTPLIB_USE_ZSTD_IF_AVAILABLE "" OFF)
    FetchContent_MakeAvailable(cpp_httplib)

    target_compile_definitions(rwkv_mobile_internal PUBLIC ENABLE_SERVER)
    if (TARGET httplib::httplib)
        target_link_libraries(rwkv_mobile_internal PUBLIC httplib::httplib)
    else()
        target_link_libraries(rwkv_mobile_internal PUBLIC httplib)
    endif()
    if (WIN32)
        target_link_libraries(rwkv_mobile_internal PUBLIC ws2_32)
    endif()
endif()

if (DEFINED ANDROID_NDK)
    find_library(log-lib log)
    find_library(android-lib android)
    target_link_libraries(rwkv_mobile_internal PUBLIC ${log-lib} ${android-lib})
endif()

if (BUILD_STATIC_LIB)
    add_library(rwkv_mobile_static STATIC src/c_api.cpp)
    target_include_directories(rwkv_mobile_static PUBLIC src)
    target_link_libraries(rwkv_mobile_static PRIVATE rwkv_mobile_internal)

    if (IOS)
        target_compile_definitions(rwkv_mobile_static PUBLIC PLATFORM_IS_IOS)
    endif()

    set (STATIC_LIBS_LIST $<TARGET_FILE:rwkv_mobile_static> $<TARGET_FILE:rwkv_mobile_internal>)

    if (ENABLE_WEBRWKV_BACKEND)
        target_link_libraries(rwkv_mobile_static PRIVATE ${WEBRWKV_FFI_LIB} ${WEBRWKV_EXTRA_LIBS})
        if (IOS)
            set (STATIC_LIBS_LIST ${STATIC_LIBS_LIST} ${WEBRWKV_FFI_LIB})
        else()
            set (STATIC_LIBS_LIST ${STATIC_LIBS_LIST} ${CMAKE_BINARY_DIR}/Release/cargo/build/${Rust_CARGO_TARGET}/release/libweb_rwkv_ffi.a)
        endif()
        if (NOT IOS)
            set (STATIC_LIBS_LIST ${STATIC_LIBS_LIST} $<TARGET_FILE:bz2_static>)
        endif()
    endif()

    if (ENABLE_LLAMACPP_BACKEND)
        set (STATIC_LIBS_LIST ${STATIC_LIBS_LIST} $<TARGET_FILE:llama> $<TARGET_FILE:ggml> $<TARGET_FILE:ggml-base> $<TARGET_FILE:ggml-cpu>)
        if (GGML_METAL)
            target_link_libraries(rwkv_mobile_static PRIVATE ${LLAMACPP_EXTRA_LIBS})
            set (STATIC_LIBS_LIST ${STATIC_LIBS_LIST} $<TARGET_FILE:ggml-metal> $<TARGET_FILE:ggml-blas>)
        endif()
        if (GGML_VULKAN AND TARGET ggml-vulkan)
            target_link_libraries(rwkv_mobile_static PRIVATE ${LLAMACPP_EXTRA_LIBS})
            set (STATIC_LIBS_LIST ${STATIC_LIBS_LIST} $<TARGET_FILE:ggml-vulkan>)
        endif()
    endif()

    # if (ENABLE_NCNN_BACKEND)
    #     set (STATIC_LIBS_LIST ${STATIC_LIBS_LIST} $<TARGET_FILE:ncnn>)
    # endif()

    if (ENABLE_COREML_BACKEND)
        set (STATIC_LIBS_LIST ${STATIC_LIBS_LIST} $<TARGET_FILE:rwkv_coreml>)
    endif()

    if (ENABLE_WHISPER OR ENABLE_TTS)
        set (STATIC_LIBS_LIST ${STATIC_LIBS_LIST}
            # $<TARGET_FILE:kaldi-native-fbank-core>
            $<TARGET_FILE:kaldifst_core>
            $<TARGET_FILE:fst>
        )
        if (IOS)
            set (STATIC_LIBS_LIST ${STATIC_LIBS_LIST} $<TARGET_FILE:MNN>)
        endif()
    endif()

    add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/librwkv_mobile.a
        COMMAND libtool -static -o ${CMAKE_BINARY_DIR}/librwkv_mobile.a ${STATIC_LIBS_LIST}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Combining static libraries for rwkv_mobile"
    )
    add_custom_target(rwkv_mobile ALL DEPENDS ${CMAKE_BINARY_DIR}/librwkv_mobile.a)
else()
    add_library(rwkv_mobile SHARED src/c_api.cpp)
    if (CMAKE_ANDROID_ARCH_ABI STREQUAL "arm64-v8a")
        target_link_options(rwkv_mobile PRIVATE
            "-Wl,-z,max-page-size=16384"
            "-Wl,-z,common-page-size=16384"
        )
    endif()
    target_link_libraries(rwkv_mobile PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,rwkv_mobile_internal>")
    if (WIN32)
        target_compile_definitions(rwkv_mobile PRIVATE _DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)
    endif()
    if (ANDROID)
        target_link_libraries(rwkv_mobile PRIVATE -static-openmp)
    endif()
    if (ENABLE_WEBRWKV_BACKEND)
        target_link_libraries(rwkv_mobile PRIVATE ${WEBRWKV_FFI_LIB} ${WEBRWKV_EXTRA_LIBS})
    endif()
endif()
