set( CMAKE_SYSTEM_NAME Windows )
set( CMAKE_SYSTEM_PROCESSOR arm64 )

set( target arm64-pc-windows-msvc )

set( CMAKE_C_COMPILER    clang )
set( CMAKE_CXX_COMPILER  clang++ )

set( CMAKE_C_COMPILER_TARGET   ${target} )
set( CMAKE_CXX_COMPILER_TARGET ${target} )

set( arch_c_flags "-march=armv8.7-a -fvectorize -ffp-model=fast -fno-finite-math-only" )
set( warn_c_flags "-Wno-format -Wno-unused-variable -Wno-unused-function -Wno-gnu-zero-variadic-macro-arguments" )

# Use dynamic CRT on Windows (equivalent to /MD) to match prebuilt libs like ncnn.
set( msvc_runtime_flags "-D_DLL -D_MT -Xclang --dependent-lib=msvcrt" )

# Ensure debug CRT symbols (_CrtDbgReport, _malloc_dbg, etc.) are resolved on Windows ARM64.
set( msvc_runtime_flags_debug "-Xclang --dependent-lib=ucrtd -Xclang --dependent-lib=vcruntimed" )

set( CMAKE_C_FLAGS_INIT   "${arch_c_flags} ${warn_c_flags} ${msvc_runtime_flags}" )
set( CMAKE_CXX_FLAGS_INIT "${arch_c_flags} ${warn_c_flags} ${msvc_runtime_flags}" )

set( CMAKE_C_FLAGS_DEBUG   "${CMAKE_C_FLAGS_DEBUG} ${msvc_runtime_flags_debug}" )
set( CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${msvc_runtime_flags_debug}" )
