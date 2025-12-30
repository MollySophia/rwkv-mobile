#ifndef WHISPER_ENCODER_H
#define WHISPER_ENCODER_H

#include "multimodal/multimodal_encoder.h"
#include "whisper.h"
#include <memory>
#include <functional>

namespace rwkvmobile {

class WhisperEncoder : public MultimodalEncoder {
public:
    WhisperEncoder();
    ~WhisperEncoder() override;

    int load_model(const std::string &model_path, const std::string &adapter_path) override;
    bool encode(const std::string &path, std::vector<float> &embeddings, int &n_tokens, bool force_no_postnorm = false) override;

private:
    std::unique_ptr<whisper_context, std::function<void(whisper_context*)>> whisper_encoder_ptr;
};

} // namespace rwkvmobile

#endif // WHISPER_ENCODER_H
