#include <iostream>
#include <chrono>
#include <random>
#include <vector>
#include <string>

#include "commondef.h"
#include "runtime.h"

#define ENSURE_SUCCESS_OR_LOG_EXIT(x, msg) if (x != rwkvmobile::RWKV_SUCCESS) { std::cout << msg << std::endl; return 1; }

int main(int argc, char **argv) {
    // set stdout to be unbuffered
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <model_file> <backend> <batch_size|all>" << std::endl;
        return 1;
    }

    rwkvmobile::Runtime runtime;
    int model_id = runtime.load_model(argv[1], argv[2], "", nullptr);
    ENSURE_SUCCESS_OR_LOG_EXIT(model_id < 0 ? model_id : rwkvmobile::RWKV_SUCCESS, "Failed to load model");
    if (model_id < 0) return 1;

    int vocab_size = runtime.get_vocab_size(model_id);

    std::vector<int> batch_sizes;
    if (std::string(argv[3]) == "all") {
        batch_sizes = runtime.get_supported_batch_sizes(model_id);
        if (batch_sizes.empty()) {
            batch_sizes.push_back(1);
        }
    } else {
        batch_sizes.push_back(atoi(argv[3]));
    }

    std::cout << "Supported batch sizes:";
    for (int batch_size : runtime.get_supported_batch_sizes(model_id)) {
        std::cout << " " << batch_size;
    }
    std::cout << std::endl;

    for (int batch_size : batch_sizes) {
        rwkvmobile::Tensor1D logits;
        std::vector<int> ids(batch_size);
        for (int i = 0; i < 128; i++) {
            for (int j = 0; j < batch_size; j++) {
                ids[j] = rand() % vocab_size;
            }
            int ret = rwkvmobile::RWKV_SUCCESS;
            if (batch_size == 1) {
                ret = runtime.eval_logits(model_id, ids[0], logits);
            } else {
                ret = runtime.eval_logits_batch_decode(model_id, ids, logits);
            }
            ENSURE_SUCCESS_OR_LOG_EXIT(ret, "Failed to eval batch decode");
        }
        std::cout << "Decode speed " << "(bsz = " << batch_size << "): "
                  << runtime.get_avg_decode_speed(model_id) << " tokens/s" << std::endl;
    }

    runtime.release();

    return 0;
}
