function(rwkv_mobile_patch_llama_cpp_ohos_affinity source_file)
    if (NOT EXISTS "${source_file}")
        message(FATAL_ERROR
            "Pinned llama.cpp ggml CPU source is missing: ${source_file}")
    endif()

    file(READ "${source_file}" source_text)
    set(patched_text "${source_text}")

    string(REPLACE
        "#elif defined(__gnu_linux__)\n// TODO: this may not work on BSD, to be verified"
        "#elif defined(__gnu_linux__) || defined(__OHOS__)\n// TODO: this may not work on BSD, to be verified"
        patched_text
        "${patched_text}"
    )
    string(REPLACE
        "#ifdef __ANDROID__\n    err = sched_setaffinity(0, sizeof(cpuset), &cpuset);"
        "#if defined(__ANDROID__) || defined(__OHOS__)\n    err = sched_setaffinity(0, sizeof(cpuset), &cpuset);"
        patched_text
        "${patched_text}"
    )

    string(FIND "${patched_text}"
        "#elif defined(__gnu_linux__) || defined(__OHOS__)" platform_index)
    string(FIND "${patched_text}"
        "#if defined(__ANDROID__) || defined(__OHOS__)\n    err = sched_setaffinity"
        syscall_index)
    if (platform_index EQUAL -1 OR syscall_index EQUAL -1)
        message(FATAL_ERROR
            "Pinned llama.cpp affinity source no longer matches the reviewed OHOS patch")
    endif()

    if (NOT patched_text STREQUAL source_text)
        file(WRITE "${source_file}" "${patched_text}")
        message(STATUS
            "Enabled real per-thread sched_setaffinity support in pinned llama.cpp for OHOS")
    endif()
endfunction()
