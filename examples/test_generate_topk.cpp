#include <iostream>
#include <chrono>

#ifndef _WIN32
#include <unistd.h>
#endif

#include "commondef.h"
#include "runtime.h"
#include "c_api.h"

#define ENSURE_SUCCESS_OR_LOG_EXIT(x, msg) if (x != rwkvmobile::RWKV_SUCCESS) { std::cout << msg << std::endl; return 1; }

void callback(const char *msg, const int, const char *next) {
    std::cout << next;
}

int main(int argc, char **argv) {
    // set stdout to be unbuffered
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0] << " <vocab_file> <model_file> <backend> <prompt> <top_k>" << std::endl;
        return 1;
    }

    void *extra_data = nullptr;

#ifndef _WIN32
    std::string path;
    if (strcmp(argv[3], "qnn") == 0) {
        char *buffer;
        if ((buffer = getcwd(NULL, 0)) == NULL) {
            perror("getcwd error");
        }
        path = std::string(buffer);
        setenv("LD_LIBRARY_PATH", path.c_str(), 1);
        setenv("ADSP_LIBRARY_PATH", path.c_str(), 1);
        if (buffer) {
            free(buffer);
        }
        std::cout << "cwd: " << path << std::endl;
        path = path + "/libQnnHtp.so";
        extra_data = (void *)path.c_str();
    }
#endif

    rwkvmobile_runtime_t runtime = rwkvmobile_runtime_init();
    int model_id = rwkvmobile_runtime_load_model_with_extra(runtime, argv[2], argv[3], argv[1], extra_data);
    ENSURE_SUCCESS_OR_LOG_EXIT(model_id < 0 ? model_id : rwkvmobile::RWKV_SUCCESS, "Failed to load model");
    if (model_id < 0) return 1;

    std::cout << "Generating demo text..." << std::endl;

    std::string prompt = argv[4];
    int top_k = atoi(argv[5]);
    const char ** candidate_output_texts = rwkvmobile_runtime_gen_completion_singletoken_topk(runtime, model_id, prompt.c_str(), top_k);
    if (candidate_output_texts == nullptr) {
        std::cerr << "Failed to generate completion" << std::endl;
        return 1;
    }
    for (int i = 0; i < top_k; i++) {
        std::cout << "Candidate " << i << ": " << candidate_output_texts[i] << std::endl;
    }

    rwkvmobile_runtime_release(runtime);

    return 0;
}
