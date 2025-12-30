#include <iostream>
#include <chrono>

#include "commondef.h"
#include "runtime.h"

#define ENSURE_SUCCESS_OR_LOG_EXIT(x, msg) if (x != rwkvmobile::RWKV_SUCCESS) { std::cout << msg << std::endl; return 1; }

int main(int argc, char **argv) {
    // set stdout to be unbuffered
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <vocab_file> <model_file> <backend>" << std::endl;
        return 1;
    }

    rwkvmobile::Runtime runtime;
    int model_id = runtime.load_model(argv[2], argv[3], argv[1], nullptr);
    ENSURE_SUCCESS_OR_LOG_EXIT(model_id < 0 ? model_id : rwkvmobile::RWKV_SUCCESS, "Failed to load model");
    if (model_id < 0) return 1;

    runtime.set_prompt(model_id, "User: Hello!\n\nAssistant: Hi!\n\n");

    std::vector<std::string> input_list = {
        "Hello!"
    };
    runtime.chat(model_id, input_list, 50, nullptr);
    std::cout << "Response: " << runtime.get_response_buffer_content(model_id) << std::endl;

    runtime.clear_state(model_id);

    input_list = {
        "Hi!"
    };
    runtime.chat(model_id, input_list, 50, nullptr);
    std::cout << "Response: " << runtime.get_response_buffer_content(model_id) << std::endl;

    std::cout << std::endl;

    return 0;
}
