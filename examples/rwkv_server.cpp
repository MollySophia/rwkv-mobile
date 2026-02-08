#include <iostream>
#include <string>

#include "commondef.h"
#include "c_api.h"

struct ServerArgs {
    std::string host = "0.0.0.0";
    int port = 8000;
    int threads = 0;
    std::string model_path;
    std::string tokenizer_path;
    std::string backend_name;
    std::string model_name = "rwkv";
    int default_max_tokens = 256;
    float temperature = 1.0f;
    int top_k = 1;
    float top_p = 1.0f;
    float presence_penalty = 0.0f;
    float frequency_penalty = 0.0f;
    float penalty_decay = 0.0f;
    bool has_temperature = false;
    bool has_top_k = false;
    bool has_top_p = false;
    bool has_presence_penalty = false;
    bool has_frequency_penalty = false;
    bool has_penalty_decay = false;
};

static void print_usage(const char * argv0) {
    std::cout
        << "Usage: " << argv0
        << " --model <path> --tokenizer <path> --backend <name> [options]\n"
        << "Options:\n"
        << "  --host <ip>                 Bind host (default 0.0.0.0)\n"
        << "  --port <port>               Bind port (default 8000)\n"
        << "  --threads <n>               HTTP worker threads (default: cpu count)\n"
        << "  --model-name <name>         Model name in responses (default: rwkv)\n"
        << "  --max-tokens <n>            Default max_tokens (default 256)\n"
        << "  --temperature <f>           Default temperature\n"
        << "  --top-k <n>                 Default top_k\n"
        << "  --top-p <f>                 Default top_p\n"
        << "  --presence-penalty <f>      Default presence_penalty\n"
        << "  --frequency-penalty <f>     Default frequency_penalty\n"
        << "  --penalty-decay <f>         Default penalty_decay\n";
}

static bool parse_int(const std::string & value, int & out) {
    try {
        size_t idx = 0;
        int v = std::stoi(value, &idx);
        if (idx != value.size()) {
            return false;
        }
        out = v;
        return true;
    } catch (...) {
        return false;
    }
}

static bool parse_float(const std::string & value, float & out) {
    try {
        size_t idx = 0;
        float v = std::stof(value, &idx);
        if (idx != value.size()) {
            return false;
        }
        out = v;
        return true;
    } catch (...) {
        return false;
    }
}

static bool parse_args(int argc, char ** argv, ServerArgs & cfg) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            return false;
        }
        auto next_value = [&](std::string & dst) -> bool {
            if (i + 1 >= argc) {
                return false;
            }
            dst = argv[++i];
            return true;
        };
        if (arg == "--model") {
            if (!next_value(cfg.model_path)) return false;
        } else if (arg == "--tokenizer") {
            if (!next_value(cfg.tokenizer_path)) return false;
        } else if (arg == "--backend") {
            if (!next_value(cfg.backend_name)) return false;
        } else if (arg == "--host") {
            if (!next_value(cfg.host)) return false;
        } else if (arg == "--model-name") {
            if (!next_value(cfg.model_name)) return false;
        } else if (arg == "--port") {
            std::string value;
            if (!next_value(value)) return false;
            if (!parse_int(value, cfg.port)) return false;
        } else if (arg == "--threads") {
            std::string value;
            if (!next_value(value)) return false;
            if (!parse_int(value, cfg.threads)) return false;
        } else if (arg == "--max-tokens") {
            std::string value;
            if (!next_value(value)) return false;
            if (!parse_int(value, cfg.default_max_tokens)) return false;
        } else if (arg == "--temperature") {
            std::string value;
            if (!next_value(value)) return false;
            if (!parse_float(value, cfg.temperature)) return false;
            cfg.has_temperature = true;
        } else if (arg == "--top-k") {
            std::string value;
            if (!next_value(value)) return false;
            if (!parse_int(value, cfg.top_k)) return false;
            cfg.has_top_k = true;
        } else if (arg == "--top-p") {
            std::string value;
            if (!next_value(value)) return false;
            if (!parse_float(value, cfg.top_p)) return false;
            cfg.has_top_p = true;
        } else if (arg == "--presence-penalty") {
            std::string value;
            if (!next_value(value)) return false;
            if (!parse_float(value, cfg.presence_penalty)) return false;
            cfg.has_presence_penalty = true;
        } else if (arg == "--frequency-penalty") {
            std::string value;
            if (!next_value(value)) return false;
            if (!parse_float(value, cfg.frequency_penalty)) return false;
            cfg.has_frequency_penalty = true;
        } else if (arg == "--penalty-decay") {
            std::string value;
            if (!next_value(value)) return false;
            if (!parse_float(value, cfg.penalty_decay)) return false;
            cfg.has_penalty_decay = true;
        } else {
            return false;
        }
    }
    return !cfg.model_path.empty() && !cfg.tokenizer_path.empty() && !cfg.backend_name.empty();
}

int main(int argc, char ** argv) {
    ServerArgs args;
    if (!parse_args(argc, argv, args)) {
        print_usage(argv[0]);
        return 1;
    }

    rwkvmobile_runtime_t runtime = rwkvmobile_runtime_init();
    int model_id = rwkvmobile_runtime_load_model(runtime, args.model_path.c_str(), args.backend_name.c_str(), args.tokenizer_path.c_str());
    if (model_id < 0) {
        std::cerr << "Failed to load model: " << model_id << std::endl;
        rwkvmobile_runtime_release(runtime);
        return 1;
    }

    rwkvmobile_server_config config = rwkvmobile_server_config_default();
    config.host = args.host.c_str();
    config.port = args.port;
    config.threads = args.threads;
    config.model_name = args.model_name.c_str();
    config.default_max_tokens = args.default_max_tokens;
    config.temperature = args.temperature;
    config.top_k = args.top_k;
    config.top_p = args.top_p;
    config.presence_penalty = args.presence_penalty;
    config.frequency_penalty = args.frequency_penalty;
    config.penalty_decay = args.penalty_decay;
    config.has_temperature = args.has_temperature ? 1 : 0;
    config.has_top_k = args.has_top_k ? 1 : 0;
    config.has_top_p = args.has_top_p ? 1 : 0;
    config.has_presence_penalty = args.has_presence_penalty ? 1 : 0;
    config.has_frequency_penalty = args.has_frequency_penalty ? 1 : 0;
    config.has_penalty_decay = args.has_penalty_decay ? 1 : 0;

    rwkvmobile_server_t server = rwkvmobile_server_start(runtime, model_id, &config);
    if (server == nullptr) {
        std::cerr << "Failed to start server (ENABLE_SERVER may be OFF)" << std::endl;
        rwkvmobile_runtime_release(runtime);
        return 1;
    }

    rwkvmobile_server_wait(server);
    rwkvmobile_server_release(server);
    rwkvmobile_runtime_release(runtime);
    return 0;
}
