#ifndef MULTIMODAL_ENCODER_H
#define MULTIMODAL_ENCODER_H

#include <string>
#include <vector>

namespace rwkvmobile {

class MultimodalEncoder {
public:
    virtual ~MultimodalEncoder() = default;
    virtual int load_model(const std::string &model_path, const std::string &adapter_path) = 0;
    virtual bool encode(const std::string &path, std::vector<float> &embeddings, int &n_tokens, bool force_no_postnorm = false) = 0;
};

} // namespace rwkvmobile

#endif // MULTIMODAL_ENCODER_H
