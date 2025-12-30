#include "whisper_encoder.h"
#include "audio.h"
#include "commondef.h"
#include "logger.h"
#include <vector>

namespace rwkvmobile {

WhisperEncoder::WhisperEncoder() : whisper_encoder_ptr(nullptr, [](whisper_context* p) { if (p) whisper_free(p); }) {}

WhisperEncoder::~WhisperEncoder() = default;

int WhisperEncoder::load_model(const std::string &model_path, const std::string &adapter_path) {
    whisper_context_params cparams = whisper_context_default_params();
    whisper_encoder_ptr.reset(whisper_init_from_file_with_params(model_path.c_str(), cparams));
    if (whisper_encoder_ptr == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    whisper_init_state(whisper_encoder_ptr.get());
    return RWKV_SUCCESS;
}

bool WhisperEncoder::encode(const std::string &path, std::vector<float> &embeddings, int &n_tokens, bool force_no_postnorm) {
    wav_file wav;
    if (!wav.load(path)) {
        LOGE("Failed to load wav file from %s", path.c_str());
        return false;
    }

    whisper_pcm_to_mel(whisper_encoder_ptr.get(), wav.samples.data(), wav.samples.size(), 4);
    whisper_encode(whisper_encoder_ptr.get(), 0, 4);
    auto embd = whisper_get_adapter_output_tensor(whisper_encoder_ptr.get());
    if (embd == nullptr) {
        return false;
    }
    n_tokens = embd->ne[1];
    int embedding_dim = embd->ne[0];
    size_t embedding_size = (size_t)n_tokens * embedding_dim;
    const float* embedding_data = (const float*)embd->data;
    embeddings.assign(embedding_data, embedding_data + embedding_size);

    return true;
}

} // namespace rwkvmobile
