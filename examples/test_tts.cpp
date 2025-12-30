#include <iostream>
#include <fstream>
#include <chrono>

#include "commondef.h"
#include "runtime.h"
#include "c_api.h"
#include "logger.h"

int main(int argc, char **argv) {
    // set stdout to be unbuffered
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc != 5 && argc != 6) {
        std::cerr << "Usage: " << argv[0] << " <model_file> <backend> <tokenizer_file> <wav_file> <enable_cache>" << std::endl;
        return 1;
    }
    rwkvmobile::Runtime runtime;
    int model_id = runtime.load_model(argv[1], argv[2], argv[3], nullptr);
    if (model_id < 0) return 1;

    bool enable_cache = false;
    if (argc == 6) {
        enable_cache = std::stoi(argv[5]);
    }

    runtime.sparktts_load_models(
        "wav2vec2-large-xlsr-53.mnn",
        "BiCodecTokenize.mnn",
        "BiCodecDetokenize.mnn"
    );

    if (enable_cache) {
        runtime.set_cache_dir("./");
    }

    for (int i = 0; i < 1; i++) {
        runtime.run_spark_tts_zeroshot_streaming(model_id, "他们小心翼翼地调整电路，确保每个部件都正确连接，红灯、绿灯、黄灯依次亮起，仿佛在讲述一个关于交通规则的故事。", "", argv[4], "output.wav");
    }
    // runtime.run_spark_tts("他们小心翼翼地调整电路，确保每个部件都正确连接，红灯、绿灯、黄灯依次亮起，仿佛在讲述一个关于交通规则的故事。", "", argv[4], "output.wav");

    runtime.release();

    return 0;
}
