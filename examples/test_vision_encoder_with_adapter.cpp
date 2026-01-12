#include <iostream>
#if _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

#include "commondef.h"
#include "runtime.h"
#include "c_api.h"

#define ENSURE_SUCCESS_OR_LOG_EXIT(x, msg) if (x != rwkvmobile::RWKV_SUCCESS) { std::cout << msg << std::endl; return 1; }

void custom_sleep(int seconds) {
#if _WIN32
    Sleep(seconds * 1000);
#else
    sleep(seconds);
#endif
}

char msg0[300];

int main(int argc, char **argv) {
    // set stdout to be unbuffered
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc != 7) {
        std::cerr << "Usage: " << argv[0] << " <model_file> <encoder_file> <adapter_file> <tokenizer_file> <image_file> <backend>" << std::endl;
        return 1;
    }

    rwkvmobile_runtime_t runtime = rwkvmobile_runtime_init();
    int model_id = rwkvmobile_runtime_load_model(runtime, argv[1], argv[6], argv[4]);
    ENSURE_SUCCESS_OR_LOG_EXIT(model_id < 0 ? model_id : rwkvmobile::RWKV_SUCCESS, "Failed to load model");
    rwkvmobile_runtime_load_vision_encoder_and_adapter(runtime, model_id, argv[2], argv[3]);
    rwkvmobile_runtime_set_sampler_params(runtime, model_id, {1.0, 1, 1.0});
    rwkvmobile_runtime_set_penalty_params(runtime, model_id, {0.0, 0.0, 0.0});
    rwkvmobile_runtime_set_eos_token(runtime, model_id, "\x17");
    rwkvmobile_runtime_set_bos_token(runtime, model_id, "\x16");
    rwkvmobile_runtime_set_token_banned(runtime, model_id, {0}, 1);
    rwkvmobile_runtime_set_space_after_roles(runtime, model_id, 0);

    const char *unique_identifier = "abababababa";
    rwkvmobile_runtime_set_image_unique_identifier(runtime, unique_identifier);

    snprintf(msg0, sizeof(msg0), "<%s>%s</%s>Recognize text", unique_identifier, argv[5], unique_identifier);
    const char *input_list[] = {msg0};

    rwkvmobile_runtime_eval_chat_with_history_async(runtime, model_id, input_list, 1, 500, nullptr, false, false, true, FORCE_LANG_NONE);

    while (rwkvmobile_runtime_is_generating(runtime, model_id)) {
        custom_sleep(1);
    }

    struct response_buffer buffer = rwkvmobile_runtime_get_response_buffer_content(runtime, model_id);
    std::cout << "\nResponse:\n" << buffer.content << std::endl;
    rwkvmobile_runtime_free_response_buffer(buffer);

    rwkvmobile_runtime_release_vision_encoder(runtime, model_id);

    rwkvmobile_runtime_release(runtime);

    return 0;
}
