#include "vision_encoder.h"
#include "llava.h"
#include "commondef.h"
#include "logger.h"
#include <vector>
#include <cmath>

namespace rwkvmobile {

VisionEncoder::VisionEncoder() : vision_encoder_ptr(nullptr, [](clip_ctx* p) { if (p) clip_free(p); }) {}

VisionEncoder::~VisionEncoder() = default;

int VisionEncoder::load_model(const std::string &model_path, const std::string &adapter_path) {
    auto adapter_path_cstr = adapter_path.empty() ? NULL : adapter_path.c_str();
    vision_encoder_ptr.reset(clip_model_load(model_path.c_str(), adapter_path_cstr, 0));
    if (vision_encoder_ptr == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    return RWKV_SUCCESS;
}

bool VisionEncoder::encode(const std::string &path, std::vector<float> &embeddings, int &n_tokens, bool force_no_postnorm) {
    unsigned char* image_bytes;
    long image_bytes_length;
    auto loaded = load_file_to_bytes(path.c_str(), &image_bytes, &image_bytes_length);
    if (!loaded) {
        LOGE("Failed to load image file from %s", path.c_str());
        return false;
    }

    image_u8 img;
    if (!image_u8_load_from_bytes(image_bytes, image_bytes_length, img)) {
        LOGE("Failed to load image from bytes: %s", path.c_str());
        free(image_bytes);
        return false;
    }
    std::vector<image_f32> img_batch;
    preprocess(img, img_batch);

    LOGI("Image size: %d x %d, buffer size: %zu", img_batch[0].nx, img_batch[0].ny, img_batch[0].buf.size());

    auto embd = llava_image_embed_make_with_bytes(vision_encoder_ptr.get(), 4, image_bytes, image_bytes_length, force_no_postnorm);
    free(image_bytes);
    if (embd == nullptr) {
        LOGE("Failed to embed image from %s", path.c_str());
        return false;
    }
    n_tokens = embd->n_image_pos;
    int embedding_dim = clip_n_mmproj_embd(vision_encoder_ptr.get());
    size_t embedding_size = (size_t)n_tokens * embedding_dim;
    embeddings.assign(embd->embed, embd->embed + embedding_size);
    llava_image_embed_free(embd);
    return true;
}

} // namespace rwkvmobile
