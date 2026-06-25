#ifndef VISION_ENCODER_H
#define VISION_ENCODER_H

#include "multimodal/multimodal_encoder.h"
// #include "clip.h"
#include <memory>
#include <functional>

#include "MNN/Interpreter.hpp"
#include <MNN/expr/Module.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>

namespace rwkvmobile {

// RGB uint8 image
struct image_u8 {
    int nx;
    int ny;

    std::vector<uint8_t> buf;
};

// RGB float32 image (NHWC)
// Memory layout: RGBRGBRGB...
struct image_f32 {
    int nx;
    int ny;

    std::vector<float> buf;
};

struct qwen_vl_grid {
    int t = 1;
    int h = 0;
    int w = 0;
};

class VisionEncoder : public MultimodalEncoder {
public:
    VisionEncoder();
    ~VisionEncoder() override;

    int max_image_size = 768;
    int split_image_size = 384;
    bool resize_to_max_side_len = true;
    bool qwen_vl_mode = false;
    bool qwen_combined_model = false;
    bool qwen_encoder_module_mode = false;
    int qwen_vl_min_pixels = 65536;
    int qwen_vl_max_pixels = 589824;
    int qwen_last_num_patches = -1;
    int qwen_last_num_tokens = -1;

    int last_batch_size = -1;

    float image_mean[3] = {0.5, 0.5, 0.5};
    float image_std[3] = {0.5, 0.5, 0.5};

    int load_model(const std::string &model_path, const std::string &adapter_path) override;
    bool encode(const std::string &path, std::vector<float> &embeddings, int &n_tokens, bool force_no_postnorm = false) override;
    std::vector<int> prefix_tokens() const override;
    std::vector<int> suffix_tokens() const override;

private:
    // std::unique_ptr<clip_ctx, std::function<void(clip_ctx*)>> vision_encoder_ptr;
    MNN::Interpreter *vision_encoder_mnn_interpretor = nullptr;
    MNN::Session *vision_encoder_mnn_session = nullptr;
    MNN::Interpreter *vision_adapter_mnn_interpretor = nullptr;
    MNN::Session *vision_adapter_mnn_session = nullptr;
    std::shared_ptr<MNN::Express::Module> vision_encoder_mnn_module;

    MNN::RuntimeInfo mnn_runtime;

    void preprocess(const image_u8 &img, std::vector<image_f32> &res_imgs);
    void preprocess_qwen_vl_patches(const image_u8 &img, std::vector<float> &patches, qwen_vl_grid &grid);
    void smart_resize_qwen_vl(int height, int width, int factor, int min_pixels, int max_pixels, int &resized_height, int &resized_width);

    void bilinear_resize(const image_u8& src, image_u8& dst, int target_width, int target_height);
    void bicubic_resize(const image_u8& src, image_u8& dst, int target_width, int target_height);
    void rescale_image_u8_to_f32(const image_u8* src, image_f32* dst, const double scale);
    void normalize_image_f32(const image_f32* src, image_f32* dst, const float mean[3], const float std[3]);

    bool load_file_to_bytes(const char* path, unsigned char** bytesOut, long *sizeOut);
    bool image_u8_load_from_bytes(const unsigned char * bytes, size_t bytes_length, image_u8 &img);
};

} // namespace rwkvmobile

#endif // VISION_ENCODER_H
