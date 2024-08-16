#include <iostream>

#include "commondef.h"
#include "runtime.h"

#define ENSURE_SUCCESS_OR_LOG_EXIT(x, msg) if (x != rwkvmobile::RWKV_SUCCESS) { std::cout << msg << std::endl; return 1; }

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <vocab_file> <model_file>" << std::endl;
        return 1;
    }

    rwkvmobile::runtime rumtime;
    ENSURE_SUCCESS_OR_LOG_EXIT(rumtime.init("web-rwkv"), "Failed to initialize runtime");
    ENSURE_SUCCESS_OR_LOG_EXIT(rumtime.load_tokenizer(argv[1]), "Failed to load tokenizer");
    ENSURE_SUCCESS_OR_LOG_EXIT(rumtime.load_model(argv[2]), "Failed to load model");

    std::cout << "Generating demo text..." << std::endl;
    std::string response;
    ENSURE_SUCCESS_OR_LOG_EXIT(rumtime.chat("User", "Assistant", "你好！", response), "Failed to generate chat message");

    std::cout << "User: Hello!\n\nAssistant:" << response << std::endl;
}
