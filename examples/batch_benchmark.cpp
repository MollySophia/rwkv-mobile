#include <iostream>
#include <chrono>
#include <random>
#include <vector>

#include "commondef.h"
#include "runtime.h"

#define ENSURE_SUCCESS_OR_LOG_EXIT(x, msg) if (x != rwkvmobile::RWKV_SUCCESS) { std::cout << msg << std::endl; return 1; }

int main(int argc, char **argv) {
    // set stdout to be unbuffered
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <model_file> <backend> <batch_size>" << std::endl;
        return 1;
    }

    rwkvmobile::Runtime runtime;
    int model_id = runtime.load_model(argv[1], argv[2], "", nullptr);
    ENSURE_SUCCESS_OR_LOG_EXIT(model_id < 0 ? model_id : rwkvmobile::RWKV_SUCCESS, "Failed to load model");
    if (model_id < 0) return 1;

    int batch_size = atoi(argv[3]);

    int vocab_size = runtime.get_vocab_size(model_id);

    rwkvmobile::Tensor1D logits;

    std::vector<int> ids(batch_size);
    for (int i = 0; i < 128; i++) {
        for (int j = 0; j < batch_size; j++) {
            ids[j] = rand() % vocab_size;
        }
        if (batch_size == 1)
            runtime.eval_logits(model_id, ids[0], logits);
        else
            runtime.eval_logits_batch_decode(model_id, ids, logits);
    }
    std::cout << "Decode speed " << "(bsz = " << batch_size << "): " << runtime.get_avg_decode_speed(model_id) << " tokens/s" << std::endl;

    runtime.release();

    return 0;
}
