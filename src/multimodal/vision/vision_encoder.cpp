#include "vision_encoder.h"
// #include "llava.h"
#include "commondef.h"
#include "logger.h"
#include <vector>
#include <cmath>
#include <filesystem>
#include <cstring>
#include <iostream>

namespace rwkvmobile {

VisionEncoder::VisionEncoder() {
    MNN::ScheduleConfig config;
    mnn_runtime = MNN::Interpreter::createRuntime({config});
}

VisionEncoder::~VisionEncoder() {
    if (vision_encoder_mnn_interpretor) {
        delete vision_encoder_mnn_interpretor;
    }
    if (vision_adapter_mnn_interpretor) {
        delete vision_adapter_mnn_interpretor;
    }
}

int VisionEncoder::load_model(const std::string &model_path, const std::string &adapter_path) {
    if (!std::filesystem::exists(model_path)) {
        LOGE("Vision encoder model file not found: %s", model_path.c_str());
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    if (!std::filesystem::exists(adapter_path)) {
        LOGE("Vision adapter model file not found: %s", adapter_path.c_str());
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    MNN::ScheduleConfig conf;
    conf.type = MNN_FORWARD_CPU;
    MNN::BackendConfig backendConfig;
    backendConfig.memory = MNN::BackendConfig::Memory_Low;
    backendConfig.power = MNN::BackendConfig::Power_High;
    backendConfig.precision = MNN::BackendConfig::Precision_Low;
    conf.backendConfig = &backendConfig;
    vision_encoder_mnn_interpretor = MNN::Interpreter::createFromFile(model_path.c_str());
    vision_encoder_mnn_session = vision_encoder_mnn_interpretor->createSession(conf, mnn_runtime);
    vision_adapter_mnn_interpretor = MNN::Interpreter::createFromFile(adapter_path.c_str());
    vision_adapter_mnn_session = vision_adapter_mnn_interpretor->createSession(conf, mnn_runtime);
    if (vision_encoder_mnn_session == nullptr || vision_adapter_mnn_session == nullptr) {
        LOGE("Failed to create session for vision encoder or adapter");
        if (vision_encoder_mnn_session != nullptr) {
            delete vision_encoder_mnn_session;
        }
        if (vision_adapter_mnn_session != nullptr) {
            delete vision_adapter_mnn_session;
        }
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

    auto pixelValTensor = vision_encoder_mnn_interpretor->getSessionInput(vision_encoder_mnn_session, "pixel_values");
    std::vector<int> input_shape = {1, 3, 384, 384};
    vision_encoder_mnn_interpretor->resizeTensor(pixelValTensor, input_shape);
    vision_encoder_mnn_interpretor->resizeSession(vision_encoder_mnn_session);

    auto adapterInputTensor = vision_adapter_mnn_interpretor->getSessionInput(vision_adapter_mnn_session, "input");
    std::vector<int> adapter_input_shape = {1, 576, 768};
    vision_adapter_mnn_interpretor->resizeTensor(adapterInputTensor, adapter_input_shape);
    vision_adapter_mnn_interpretor->resizeSession(vision_adapter_mnn_session);
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

    auto pixelValTensor = vision_encoder_mnn_interpretor->getSessionInput(vision_encoder_mnn_session, "pixel_values");
    auto nchw_tensor = new MNN::Tensor(pixelValTensor, MNN::Tensor::CAFFE);
    for (int k = 0; k < 3; k++) {
        for (int y = 0; y < img_batch[0].ny; y++) {
            for (int x = 0; x < img_batch[0].nx; x++) {
                nchw_tensor->host<float>()[k * img_batch[0].ny * img_batch[0].nx + y * img_batch[0].nx + x] = img_batch[0].buf[3 * (y * img_batch[0].nx + x) + k];
            }
        }
    }

    pixelValTensor->copyFromHostTensor(nchw_tensor);
    delete nchw_tensor;
    vision_encoder_mnn_interpretor->runSession(vision_encoder_mnn_session);
    auto encoderOutputTensor = vision_encoder_mnn_interpretor->getSessionOutput(vision_encoder_mnn_session, "last_hidden_state");
    void *encoderOutputPtr = encoderOutputTensor->map(MNN::Tensor::MAP_TENSOR_READ, encoderOutputTensor->getDimensionType());

    auto adapterInputTensor = vision_adapter_mnn_interpretor->getSessionInput(vision_adapter_mnn_session, "input");
    nchw_tensor = new MNN::Tensor(adapterInputTensor, MNN::Tensor::CAFFE);
    memcpy(nchw_tensor->host<float>(), encoderOutputPtr, encoderOutputTensor->elementSize() * sizeof(float));
    adapterInputTensor->copyFromHostTensor(nchw_tensor);
    delete nchw_tensor;
    encoderOutputTensor->unmap(MNN::Tensor::MAP_TENSOR_READ, encoderOutputTensor->getDimensionType(), encoderOutputPtr);

    std::string output_name = force_no_postnorm ? "output" : "output_with_rwkv_norm";
    auto adapterOutputTensor = vision_adapter_mnn_interpretor->getSessionOutput(vision_adapter_mnn_session, output_name.c_str());
    if (adapterOutputTensor == nullptr) {
        LOGE("Failed to get output tensor for adapter");
        return false;
    }
    vision_adapter_mnn_interpretor->runSession(vision_adapter_mnn_session);
    int output_size = adapterOutputTensor->elementSize();
    void *adapterOutputPtr = adapterOutputTensor->map(MNN::Tensor::MAP_TENSOR_READ, adapterOutputTensor->getDimensionType());
    embeddings.assign((float*)adapterOutputPtr, (float*)adapterOutputPtr + output_size);
    adapterOutputTensor->unmap(MNN::Tensor::MAP_TENSOR_READ, adapterOutputTensor->getDimensionType(), adapterOutputPtr);
    n_tokens = 576;

    return true;
}

} // namespace rwkvmobile
