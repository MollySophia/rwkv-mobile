#ifndef RWKV_HTTP_SERVER_H
#define RWKV_HTTP_SERVER_H

#include <memory>
#include <string>
#include <thread>

namespace rwkvmobile {

class Runtime;

struct HttpServerConfig {
    std::string host = "0.0.0.0";
    int port = 8000;
    int threads = 0;
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

class RwkvHttpServer {
public:
    RwkvHttpServer(Runtime * runtime, int model_id, HttpServerConfig config);
    ~RwkvHttpServer();

    int start();
    int stop();
    int wait();
    bool is_running() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace rwkvmobile

#endif // RWKV_HTTP_SERVER_H
