#include "vision_encoder.h"
#include "commondef.h"
#include "logger.h"
#include <vector>
#include <cmath>
#include <filesystem>
#include <cstring>
#include <cstdlib>
#include "soc_detect.h"

namespace rwkvmobile {

VisionEncoder::VisionEncoder() {
    if (const char *min_pixels = std::getenv("RWKV_QWEN_VL_MIN_PIXELS")) {
        qwen_vl_min_pixels = std::max(1, std::atoi(min_pixels));
    }
    if (const char *max_pixels = std::getenv("RWKV_QWEN_VL_MAX_PIXELS")) {
        qwen_vl_max_pixels = std::max(qwen_vl_min_pixels, std::atoi(max_pixels));
    }
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
    qwen_combined_model = adapter_path.empty();
    if (!qwen_combined_model && !std::filesystem::exists(adapter_path)) {
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
    if (!qwen_combined_model) {
        vision_adapter_mnn_interpretor = MNN::Interpreter::createFromFile(adapter_path.c_str());
        vision_adapter_mnn_session = vision_adapter_mnn_interpretor->createSession(conf, mnn_runtime);
    }
    if (vision_encoder_mnn_session == nullptr) {
        MNN::Express::Module::Config module_config;
        module_config.dynamic = true;
        MNN::Express::Module::BackendInfo module_backend;
        module_backend.type = MNN_FORWARD_CPU;
        module_backend.config = &backendConfig;
        module_config.backend = &module_backend;
        vision_encoder_mnn_module.reset(MNN::Express::Module::load(
            std::vector<std::string>{"pixel_values", "image_grid_thw"},
            std::vector<std::string>{"pooler_output"},
            model_path.c_str(),
            &module_config
        ));
        qwen_encoder_module_mode = vision_encoder_mnn_module != nullptr;
        qwen_vl_mode = qwen_encoder_module_mode;
        if (qwen_encoder_module_mode) {
            LOGI("Qwen-VL dynamic encoder loaded with MNN Module API");
        }
    }
    if ((!qwen_encoder_module_mode && vision_encoder_mnn_session == nullptr) || (!qwen_combined_model && vision_adapter_mnn_session == nullptr)) {
        LOGE("Failed to create session for vision encoder or adapter");
        if (vision_encoder_mnn_session != nullptr) {
            delete vision_encoder_mnn_session;
        }
        if (vision_adapter_mnn_session != nullptr) {
            delete vision_adapter_mnn_session;
        }
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

#if __ANDROID__
    auto cpu_groups = get_cpu_groups();
    vision_encoder_mnn_interpretor->setSessionHint(MNN::Interpreter::HintMode::CPU_CORE_IDS, cpu_groups[1].ids.data(), cpu_groups[1].ids.size());
    if (!qwen_combined_model) {
        vision_adapter_mnn_interpretor->setSessionHint(MNN::Interpreter::HintMode::CPU_CORE_IDS, cpu_groups[1].ids.data(), cpu_groups[1].ids.size());
    }
    std::string msg = "[Vision Encoder]: binding mnn to cpu core ids: ";
    for (int i = 0; i < cpu_groups[1].ids.size(); i++) {
        msg += std::to_string(cpu_groups[1].ids[i]) + " ";
    }
    LOGI("%s", msg.c_str());
#endif

    MNN::Tensor *pixelValTensor = nullptr;
    MNN::Tensor *gridTensor = nullptr;
    if (!qwen_encoder_module_mode) {
        const auto &encoder_inputs = vision_encoder_mnn_interpretor->getSessionInputAll(vision_encoder_mnn_session);
        auto pixel_it = encoder_inputs.find("pixel_values");
        auto grid_it = encoder_inputs.find("image_grid_thw");
        pixelValTensor = pixel_it == encoder_inputs.end() ? nullptr : pixel_it->second;
        gridTensor = grid_it == encoder_inputs.end() ? nullptr : grid_it->second;
    }
    auto filename = std::filesystem::path(model_path).filename().string();
    auto grid_pos = filename.find("grid");
    qwen_vl_mode = qwen_vl_mode || qwen_combined_model || gridTensor != nullptr || grid_pos != std::string::npos ||
        (pixelValTensor != nullptr && pixelValTensor->dimensions() == 2 && pixelValTensor->length(1) == 3 * 2 * 16 * 16);
    if (qwen_vl_mode) {
        LOGI("Qwen-VL vision loaded with dynamic grid, split_adapter=%d", qwen_combined_model ? 0 : 1);
        return RWKV_SUCCESS;
    } else {
        std::vector<int> input_shape = {1, 3, split_image_size, split_image_size};
        vision_encoder_mnn_interpretor->resizeTensor(pixelValTensor, input_shape);
        vision_encoder_mnn_interpretor->resizeSession(vision_encoder_mnn_session);
    }

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
    free(image_bytes);
    std::vector<image_f32> img_batch;
    if (qwen_vl_mode) {
        std::vector<float> patches;
        qwen_vl_grid grid;
        preprocess_qwen_vl_patches(img, patches, grid);
        const int num_patches = grid.t * grid.h * grid.w;
        const int num_tokens = num_patches / 4;
        if (grid.h % 2 != 0 || grid.w % 2 != 0 || num_patches <= 0 || num_tokens <= 0) {
            LOGE("Invalid Qwen-VL grid after preprocessing: %dx%dx%d", grid.t, grid.h, grid.w);
            return false;
        }
        LOGI("Qwen-VL preprocess grid: %dx%dx%d, patches: %d, merged tokens: %d", grid.t, grid.h, grid.w, num_patches, num_tokens);
        MNN::Tensor *pixelValTensor = nullptr;
        MNN::Tensor *gridTensor = nullptr;
        if (!qwen_encoder_module_mode) {
            const auto &encoder_inputs = vision_encoder_mnn_interpretor->getSessionInputAll(vision_encoder_mnn_session);
            auto pixel_it = encoder_inputs.find("pixel_values");
            if (pixel_it == encoder_inputs.end()) {
                LOGE("Failed to get Qwen-VL encoder pixel_values input tensor");
                return false;
            }
            auto grid_it = encoder_inputs.find("image_grid_thw");
            pixelValTensor = pixel_it->second;
            gridTensor = grid_it == encoder_inputs.end() ? nullptr : grid_it->second;
            std::vector<int> input_shape = {num_patches, 3 * 2 * 16 * 16};
            vision_encoder_mnn_interpretor->resizeTensor(pixelValTensor, input_shape);
            if (gridTensor != nullptr) {
                std::vector<int> grid_shape = {1, 3};
                vision_encoder_mnn_interpretor->resizeTensor(gridTensor, grid_shape);
            }
            vision_encoder_mnn_interpretor->resizeSession(vision_encoder_mnn_session);
        }

        if (!qwen_combined_model) {
            auto adapterInputTensor = vision_adapter_mnn_interpretor->getSessionInput(vision_adapter_mnn_session, "input");
            if (adapterInputTensor == nullptr) {
                LOGE("Failed to get Qwen-VL adapter input tensor");
                return false;
            }
            if (num_tokens != qwen_last_num_tokens) {
                std::vector<int> adapter_input_shape = {num_tokens, 1024};
                vision_adapter_mnn_interpretor->resizeTensor(adapterInputTensor, adapter_input_shape);
                vision_adapter_mnn_interpretor->resizeSession(vision_adapter_mnn_session);
            }
        }
        qwen_last_num_patches = num_patches;
        qwen_last_num_tokens = num_tokens;

        MNN::Tensor *outputTensor = nullptr;
        std::vector<float> module_encoder_output;
        if (qwen_encoder_module_mode) {
            LOGI("Qwen-VL dynamic encoder input: pixel_values=[%d,%d], image_grid_thw=[1,3]", num_patches, 3 * 2 * 16 * 16);
            auto pixel_var = MNN::Express::_Input({num_patches, 3 * 2 * 16 * 16}, MNN::Express::NCHW, halide_type_of<float>());
            memcpy(pixel_var->writeMap<float>(), patches.data(), patches.size() * sizeof(float));
            int32_t grid_data[3] = {static_cast<int32_t>(grid.t), static_cast<int32_t>(grid.h), static_cast<int32_t>(grid.w)};
            auto grid_var = MNN::Express::_Input({1, 3}, MNN::Express::NCHW, halide_type_of<int32_t>());
            memcpy(grid_var->writeMap<int32_t>(), grid_data, sizeof(grid_data));
            auto encoder_outputs = vision_encoder_mnn_module->onForward({pixel_var, grid_var});
            if (encoder_outputs.empty()) {
                LOGE("Failed to run Qwen-VL dynamic encoder module");
                return false;
            }
            auto encoder_output_var = encoder_outputs[0];
            const auto *encoder_info = encoder_output_var->getInfo();
            if (encoder_info == nullptr || encoder_info->size <= 0) {
                LOGE("Failed to get Qwen-VL dynamic encoder output info");
                return false;
            }
            const float *encoder_data = encoder_output_var->readMap<float>();
            module_encoder_output.assign(encoder_data, encoder_data + encoder_info->size);
            LOGI("Qwen-VL dynamic encoder output elements: %d", encoder_info->size);
        } else {
            auto patch_tensor = new MNN::Tensor(pixelValTensor, MNN::Tensor::TENSORFLOW);
            memcpy(patch_tensor->host<float>(), patches.data(), patches.size() * sizeof(float));
            pixelValTensor->copyFromHostTensor(patch_tensor);
            delete patch_tensor;
            if (gridTensor != nullptr) {
                auto gridHostTensor = new MNN::Tensor(gridTensor, MNN::Tensor::TENSORFLOW);
                int32_t *grid_data = gridHostTensor->host<int32_t>();
                grid_data[0] = static_cast<int32_t>(grid.t);
                grid_data[1] = static_cast<int32_t>(grid.h);
                grid_data[2] = static_cast<int32_t>(grid.w);
                gridTensor->copyFromHostTensor(gridHostTensor);
                delete gridHostTensor;
            }

            vision_encoder_mnn_interpretor->runSession(vision_encoder_mnn_session);
            if (qwen_combined_model) {
                const char *output_name = force_no_postnorm ? "image_embeddings" : "output_with_rwkv_norm";
                outputTensor = vision_encoder_mnn_interpretor->getSessionOutput(vision_encoder_mnn_session, output_name);
                if (outputTensor == nullptr) {
                    outputTensor = vision_encoder_mnn_interpretor->getSessionOutput(vision_encoder_mnn_session, "image_embeddings");
                }
                if (outputTensor == nullptr) {
                    outputTensor = vision_encoder_mnn_interpretor->getSessionOutput(vision_encoder_mnn_session, "output");
                }
            }
        }
        if (!qwen_combined_model) {
            MNN::Tensor *encoderOutputTensor = nullptr;
            if (!qwen_encoder_module_mode) {
                encoderOutputTensor = vision_encoder_mnn_interpretor->getSessionOutput(vision_encoder_mnn_session, "pooler_output");
            }
            if (!qwen_encoder_module_mode && encoderOutputTensor == nullptr) {
                encoderOutputTensor = vision_encoder_mnn_interpretor->getSessionOutput(vision_encoder_mnn_session, "output");
            }
            if (!qwen_encoder_module_mode && encoderOutputTensor == nullptr) {
                LOGE("Failed to get Qwen-VL encoder pooler output tensor");
                return false;
            }
            auto adapterInputTensor = vision_adapter_mnn_interpretor->getSessionInput(vision_adapter_mnn_session, "input");
            auto adapterTensor = new MNN::Tensor(adapterInputTensor, MNN::Tensor::TENSORFLOW);
            if (qwen_encoder_module_mode) {
                memcpy(adapterTensor->host<float>(), module_encoder_output.data(), module_encoder_output.size() * sizeof(float));
            } else {
                void *encoderOutputPtr = encoderOutputTensor->map(MNN::Tensor::MAP_TENSOR_READ, encoderOutputTensor->getDimensionType());
                memcpy(adapterTensor->host<float>(), encoderOutputPtr, encoderOutputTensor->elementSize() * sizeof(float));
                encoderOutputTensor->unmap(MNN::Tensor::MAP_TENSOR_READ, encoderOutputTensor->getDimensionType(), encoderOutputPtr);
            }
            adapterInputTensor->copyFromHostTensor(adapterTensor);
            delete adapterTensor;

            vision_adapter_mnn_interpretor->runSession(vision_adapter_mnn_session);
            const char *output_name = force_no_postnorm ? "image_embeddings" : "output_with_rwkv_norm";
            outputTensor = vision_adapter_mnn_interpretor->getSessionOutput(vision_adapter_mnn_session, output_name);
            if (outputTensor == nullptr) {
                outputTensor = vision_adapter_mnn_interpretor->getSessionOutput(vision_adapter_mnn_session, "image_embeddings");
            }
            if (outputTensor == nullptr) {
                outputTensor = vision_adapter_mnn_interpretor->getSessionOutput(vision_adapter_mnn_session, "output");
            }
        }
        if (outputTensor == nullptr) {
            LOGE("Failed to get Qwen-VL image embeddings output tensor");
            return false;
        }
        int output_size = outputTensor->elementSize();
        void *outputPtr = outputTensor->map(MNN::Tensor::MAP_TENSOR_READ, outputTensor->getDimensionType());
        embeddings.assign((float*)outputPtr, (float*)outputPtr + output_size);
        outputTensor->unmap(MNN::Tensor::MAP_TENSOR_READ, outputTensor->getDimensionType(), outputPtr);
        int dims = outputTensor->dimensions();
        int embed_dim = dims > 0 ? outputTensor->length(dims - 1) : 0;
        int token_count = 0;
        if (embed_dim > 0) {
            token_count = 1;
            for (int i = 0; i < dims - 1; i++) {
                token_count *= outputTensor->length(i);
            }
        }
        n_tokens = token_count;
        LOGI("Qwen-VL image grid: %dx%dx%d, n_tokens: %d", grid.t, grid.h, grid.w, n_tokens);
        return true;
    }

    preprocess(img, img_batch);

    int batch_size = static_cast<int>(img_batch.size());
    LOGI("image batch size: %d", batch_size);
    if (batch_size <= 0) {
        LOGE("Empty image batch after preprocessing");
        return false;
    }

    auto pixelValTensor = vision_encoder_mnn_interpretor->getSessionInput(vision_encoder_mnn_session, "pixel_values");
    if (batch_size != last_batch_size) {
        std::vector<int> input_shape = {batch_size, 3, split_image_size, split_image_size};
        vision_encoder_mnn_interpretor->resizeTensor(pixelValTensor, input_shape);
        vision_encoder_mnn_interpretor->resizeSession(vision_encoder_mnn_session);

        auto adapterInputTensor = vision_adapter_mnn_interpretor->getSessionInput(vision_adapter_mnn_session, "input");
        std::vector<int> adapter_input_shape = {batch_size, 576, 768};
        vision_adapter_mnn_interpretor->resizeTensor(adapterInputTensor, adapter_input_shape);
        vision_adapter_mnn_interpretor->resizeSession(vision_adapter_mnn_session);

        last_batch_size = batch_size;
    }

    auto nchw_tensor = new MNN::Tensor(pixelValTensor, MNN::Tensor::CAFFE);
    const int target_h = split_image_size;
    const int target_w = split_image_size;
    for (int b = 0; b < batch_size; b++) {
        const auto &img = img_batch[b];
        for (int k = 0; k < 3; k++) {
            for (int y = 0; y < target_h; y++) {
                for (int x = 0; x < target_w; x++) {
                    size_t src_index = 3 * (y * img.nx + x) + k;
                    size_t dst_index = ((b * 3 + k) * target_h + y) * target_w + x;
                    nchw_tensor->host<float>()[dst_index] = img.buf[src_index];
                }
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
    int dims = adapterOutputTensor->dimensions();
    int embed_dim = dims > 0 ? adapterOutputTensor->length(dims - 1) : 0;
    int token_count = 0;
    if (embed_dim > 0) {
        token_count = 1;
        for (int i = 0; i < dims - 1; i++) {
            token_count *= adapterOutputTensor->length(i);
        }
    }
    n_tokens = token_count;
    LOGI("image n_tokens: %d", n_tokens);

    return true;
}

std::vector<int> VisionEncoder::prefix_tokens() const {
    return qwen_vl_mode ? std::vector<int>{65530} : std::vector<int>{};
}

std::vector<int> VisionEncoder::suffix_tokens() const {
    return qwen_vl_mode ? std::vector<int>{65531} : std::vector<int>{};
}

} // namespace rwkvmobile
