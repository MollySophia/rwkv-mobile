#include <cstdint>
#if !__has_feature(objc_arc)
#error This file must be compiled with automatic reference counting enabled (-fobjc-arc)
#endif

#import "rwkv-coreml.h"
#import "rwkv_coreml_firstchunk_impl.h"
#import "rwkv_coreml_impl.h"
#import "rwkv_coreml_singlechunk_impl.h"

#import <CoreML/CoreML.h>

#include <stdlib.h>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <atomic>
#include "half.hpp"
#include "logger.h"

enum rwkv_coreml_state_mode {
    RWKV_COREML_STATE_MODE_COREML = 0,
    RWKV_COREML_STATE_MODE_TENSOR = 1,
    RWKV_COREML_STATE_MODE_WKV_COREML = 2,
};

static constexpr int kDefaultAsyncPrefillDecodeLoadThresholdMs = 5000;

static void rwkv_coreml_log(int level, NSString *format, ...) {
    va_list args;
    va_start(args, format);
    NSString *message = [[NSString alloc] initWithFormat:format arguments:args];
    va_end(args);

    const char *text = message.UTF8String;
    if (text == nullptr) text = "";
    switch (level) {
        case rwkvmobile::RWKV_LOG_LEVEL_DEBUG:
            rwkvmobile::LOGD("[CoreML] %s", text);
            break;
        case rwkvmobile::RWKV_LOG_LEVEL_WARN:
            rwkvmobile::LOGW("[CoreML] %s", text);
            break;
        case rwkvmobile::RWKV_LOG_LEVEL_ERROR:
            rwkvmobile::LOGE("[CoreML] %s", text);
            break;
        case rwkvmobile::RWKV_LOG_LEVEL_INFO:
        default:
            rwkvmobile::LOGI("[CoreML] %s", text);
            break;
    }
}

#define COREML_LOGI(...) rwkv_coreml_log(rwkvmobile::RWKV_LOG_LEVEL_INFO, __VA_ARGS__)
#define COREML_LOGW(...) rwkv_coreml_log(rwkvmobile::RWKV_LOG_LEVEL_WARN, __VA_ARGS__)
#define COREML_LOGE(...) rwkv_coreml_log(rwkvmobile::RWKV_LOG_LEVEL_ERROR, __VA_ARGS__)

struct rwkv_coreml_context {
    std::vector<const void *> model_decode;
    std::vector<const void *> model_prefill;
    std::vector<const void *> states;
    std::vector<const void *> state_wkv_tensors;
    std::vector<const void *> state_tokenshift_tensors;
    rwkv_coreml_state_mode state_mode = RWKV_COREML_STATE_MODE_COREML;
    int num_chunks = 0;
    std::atomic<int> load_done_chunks{0};
    float load_progress_reported = 0.f;
    int n_layers;
    int num_heads;
    int head_dim;
    int embd_dim;
    int vocab_size;
    std::atomic<int> prefill_seq_length{0};
    std::atomic<bool> load_prefill_async{false};
    std::atomic<bool> prefill_ready{false};
    std::atomic<bool> prefill_failed{false};
    std::thread prefill_load_thread;

    // IMPORTANT (ARC): these Objective-C objects live inside a C++ struct.
    // Without __strong, ARC will not automatically retain/release them, causing
    // leaks (and/or premature frees) when overwritten in decode/prefill loops.
    __strong id out_prefill = nil;
    __strong id out_decode = nil;

    // Exact byte sizes of CoreML state buffers (including any padding due to strides/alignment).
    std::vector<size_t> state_wkv_bytes_per_chunk;
    std::vector<size_t> state_tokenshift_bytes_per_chunk;
    size_t state_wkv_bytes = 0;
    size_t state_tokenshift_bytes = 0;
};

static void rwkv_coreml_join_prefill_loader(struct rwkv_coreml_context * ctx) {
    if (!ctx) return;
    if (ctx->prefill_load_thread.joinable()) {
        ctx->prefill_load_thread.join();
    }
}

static void rwkv_coreml_release_resources(struct rwkv_coreml_context * ctx) {
    if (!ctx) return;
    rwkv_coreml_join_prefill_loader(ctx);
    for (size_t i = 0; i < ctx->model_decode.size(); ++i) {
        if (ctx->model_decode[i]) CFRelease(ctx->model_decode[i]);
    }
    for (size_t i = 0; i < ctx->model_prefill.size(); ++i) {
        if (ctx->model_prefill[i]) CFRelease(ctx->model_prefill[i]);
    }
    for (size_t i = 0; i < ctx->states.size(); ++i) {
        if (ctx->states[i]) CFRelease(ctx->states[i]);
    }
    for (size_t i = 0; i < ctx->state_wkv_tensors.size(); ++i) {
        if (ctx->state_wkv_tensors[i]) CFRelease(ctx->state_wkv_tensors[i]);
    }
    for (size_t i = 0; i < ctx->state_tokenshift_tensors.size(); ++i) {
        if (ctx->state_tokenshift_tensors[i]) CFRelease(ctx->state_tokenshift_tensors[i]);
    }
    ctx->model_decode.clear();
    ctx->model_prefill.clear();
    ctx->states.clear();
    ctx->state_wkv_tensors.clear();
    ctx->state_tokenshift_tensors.clear();
    ctx->state_mode = RWKV_COREML_STATE_MODE_COREML;
    ctx->state_wkv_bytes_per_chunk.clear();
    ctx->state_tokenshift_bytes_per_chunk.clear();
    ctx->state_wkv_bytes = 0;
    ctx->state_tokenshift_bytes = 0;
    ctx->num_chunks = 0;
    ctx->load_done_chunks.store(0);
    ctx->load_progress_reported = 0.f;
    ctx->n_layers = 0;
    ctx->num_heads = 0;
    ctx->head_dim = 0;
    ctx->embd_dim = 0;
    ctx->vocab_size = 0;
    ctx->prefill_seq_length.store(0);
    ctx->load_prefill_async.store(false);
    ctx->prefill_ready.store(false);
    ctx->prefill_failed.store(false);
    // Release retained Objective-C objects eagerly (they are __strong).
    ctx->out_decode = nil;
    ctx->out_prefill = nil;
}

NSArray<NSNumber *> * get_shape_by_name(NSDictionary *model_inputs, NSString *name) {
    MLFeatureDescription *desc = model_inputs[name];
    if (desc.type == MLFeatureTypeMultiArray) {
        return desc.multiArrayConstraint.shape;
    }
    return nil;
}

static NSString * trim_string(NSString *value) {
    if (value == nil) return nil;
    NSString *trimmed = [value stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
    if (trimmed.length >= 2) {
        unichar first = [trimmed characterAtIndex:0];
        unichar last = [trimmed characterAtIndex:trimmed.length - 1];
        if ((first == '\'' && last == '\'') || (first == '\"' && last == '\"')) {
            return [trimmed substringWithRange:NSMakeRange(1, trimmed.length - 2)];
        }
    }
    return trimmed;
}

static bool parse_state_mode(NSString *value, rwkv_coreml_state_mode *state_mode_out) {
    NSString *mode = trim_string(value);
    if (mode == nil || mode.length == 0 || [mode isEqualToString:@"coreml"]) {
        if (state_mode_out) *state_mode_out = RWKV_COREML_STATE_MODE_COREML;
        return true;
    }
    if ([mode isEqualToString:@"tensor"]) {
        if (state_mode_out) *state_mode_out = RWKV_COREML_STATE_MODE_TENSOR;
        return true;
    }
    if ([mode isEqualToString:@"wkv-coreml"]) {
        if (state_mode_out) *state_mode_out = RWKV_COREML_STATE_MODE_WKV_COREML;
        return true;
    }
    COREML_LOGE(@"config.yaml invalid state_mode: %@", mode);
    return false;
}

static NSString * state_mode_name(rwkv_coreml_state_mode state_mode) {
    switch (state_mode) {
        case RWKV_COREML_STATE_MODE_COREML:
            return @"coreml";
        case RWKV_COREML_STATE_MODE_TENSOR:
            return @"tensor";
        case RWKV_COREML_STATE_MODE_WKV_COREML:
            return @"wkv-coreml";
    }
    return @"coreml";
}

static int env_int_or_default(const char *name, int default_value, int min_value, int max_value) {
    const char *value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') return default_value;
    char *end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    if (end == value || parsed < min_value || parsed > max_value) return default_value;
    return (int)parsed;
}

static MLModel * load_coreml_model_with_retry(
    NSURL *url_model,
    MLModelConfiguration *configuration,
    NSString *model_name,
    NSString *function_name,
    int chunk_idx,
    int num_chunks
) {
    const int max_attempts = env_int_or_default("RWKV_COREML_LOAD_RETRY_ATTEMPTS", 3, 1, 10);
    const int retry_delay_ms = env_int_or_default("RWKV_COREML_LOAD_RETRY_DELAY_MS", 500, 0, 10000);

    for (int attempt = 1; attempt <= max_attempts; ++attempt) {
        __strong MLModel *model = nil;
        __strong NSError *error = nil;
        auto start = std::chrono::steady_clock::now();
        @autoreleasepool {
            model = [MLModel modelWithContentsOfURL:url_model configuration:configuration error:&error];
        }
        auto end = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();

        if (error == nil && model != nil) {
            if (attempt == 1) {
                COREML_LOGI(@"Loaded chunk %d/%d (%@) %@: %.2f ms",
                      chunk_idx + 1, num_chunks, model_name, function_name, ms);
            } else {
                COREML_LOGI(@"Loaded chunk %d/%d (%@) %@: %.2f ms (attempt %d/%d)",
                      chunk_idx + 1, num_chunks, model_name, function_name, ms, attempt, max_attempts);
            }
            return model;
        }

        COREML_LOGE(@"Error loading %@ model %@ attempt %d/%d: %@ (%.2f ms)",
              function_name, model_name, attempt, max_attempts, error, ms);
        model = nil;
        error = nil;

        if (attempt < max_attempts && retry_delay_ms > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(retry_delay_ms));
        }
    }

    return nil;
}

static bool parse_coreml_config(NSString *config_path, NSString **basename_out, int *num_chunks_out, rwkv_coreml_state_mode *state_mode_out) {
    NSError *error = nil;
    NSString *content = [NSString stringWithContentsOfFile:config_path encoding:NSUTF8StringEncoding error:&error];
    if (error || content == nil) {
        COREML_LOGE(@"Error reading config.yaml: %@", error);
        return false;
    }
    NSString *basename = nil;
    int num_chunks = 0;
    // Legacy CoreML exports did not write state_mode; they used full Core ML state.
    rwkv_coreml_state_mode state_mode = RWKV_COREML_STATE_MODE_COREML;
    NSArray<NSString *> *lines = [content componentsSeparatedByCharactersInSet:[NSCharacterSet newlineCharacterSet]];
    for (NSString *line in lines) {
        NSString *trimmed = trim_string(line);
        if (trimmed.length == 0 || [trimmed hasPrefix:@"#"]) continue;
        NSRange colonRange = [trimmed rangeOfString:@":"];
        if (colonRange.location == NSNotFound) continue;
        NSString *key = trim_string([trimmed substringToIndex:colonRange.location]);
        NSString *value = trim_string([trimmed substringFromIndex:colonRange.location + 1]);
        if ([key isEqualToString:@"basename"]) {
            basename = value;
        } else if ([key isEqualToString:@"num_chunks"]) {
            num_chunks = [value intValue];
        } else if ([key isEqualToString:@"state_mode"]) {
            if (!parse_state_mode(value, &state_mode)) {
                return false;
            }
        }
    }
    if (basename == nil || basename.length == 0) {
        COREML_LOGE(@"config.yaml missing basename");
        return false;
    }
    if (num_chunks <= 0) {
        COREML_LOGE(@"config.yaml invalid num_chunks: %d", num_chunks);
        return false;
    }
    if (basename_out) *basename_out = basename;
    if (num_chunks_out) *num_chunks_out = num_chunks;
    if (state_mode_out) *state_mode_out = state_mode;
    return true;
}

static MLFeatureValue * multi_array_feature(MLMultiArray *array) {
    return [MLFeatureValue featureValueWithMultiArray:array];
}

static void copy_multi_array(MLMultiArray *dst, MLMultiArray *src) {
    if (dst == nil || src == nil) return;
    [src getBytesWithHandler:^(const void *src_bytes, NSInteger src_size) {
        if (src_bytes == nullptr || src_size <= 0) return;
        [dst getMutableBytesWithHandler:^(void *dst_bytes, NSInteger dst_size, NSArray<NSNumber *> *strides) {
            (void)strides;
            if (dst_bytes == nullptr || dst_size <= 0) return;
            const size_t n = std::min((size_t)src_size, (size_t)dst_size);
            if (n > 0) std::memcpy(dst_bytes, src_bytes, n);
            if ((size_t)dst_size > n) std::memset((uint8_t*)dst_bytes + n, 0, (size_t)dst_size - n);
        }];
    }];
}

static MLMultiArray * make_zero_multi_array_from_feature(NSDictionary *model_inputs, NSString *name) {
    MLFeatureDescription *desc = model_inputs[name];
    if (desc == nil || desc.type != MLFeatureTypeMultiArray || desc.multiArrayConstraint == nil) {
        return nil;
    }
    NSError *error = nil;
    MLMultiArray *array = [[MLMultiArray alloc] initWithShape:desc.multiArrayConstraint.shape
                                                     dataType:desc.multiArrayConstraint.dataType
                                                        error:&error];
    if (error || array == nil) {
        COREML_LOGE(@"Error allocating %@ state tensor: %@", name, error);
        return nil;
    }
    [array getMutableBytesWithHandler:^(void *mutableBytes, NSInteger size, NSArray<NSNumber *> *strides) {
        (void)strides;
        if (mutableBytes != nullptr && size > 0) std::memset(mutableBytes, 0, (size_t)size);
    }];
    return array;
}

static void set_retained_object(std::vector<const void *> &objects, int idx, id object) {
    if (idx < 0) return;
    if ((size_t)idx >= objects.size()) {
        objects.resize((size_t)idx + 1, nullptr);
    }
    if (objects[(size_t)idx] != nullptr) {
        CFRelease(objects[(size_t)idx]);
        objects[(size_t)idx] = nullptr;
    }
    if (object != nil) {
        objects[(size_t)idx] = CFBridgingRetain(object);
    }
}

static bool install_decode_chunk(struct rwkv_coreml_context *ctx, MLModel *mlmodel_decode, int chunk_idx) {
    if (ctx->state_mode == RWKV_COREML_STATE_MODE_COREML && ctx->num_chunks == 1) {
        rwkv_coreml_singlechunk_impl *model_decode = [[rwkv_coreml_singlechunk_impl alloc] initWithMLModel:mlmodel_decode];
        set_retained_object(ctx->model_decode, chunk_idx, model_decode);
        rwkv_coreml_singlechunk_implState *state = [model_decode newState];
        set_retained_object(ctx->states, chunk_idx, state);
        return true;
    }
    if (ctx->state_mode == RWKV_COREML_STATE_MODE_COREML && chunk_idx == 0) {
        rwkv_coreml_firstchunk_impl *model_decode = [[rwkv_coreml_firstchunk_impl alloc] initWithMLModel:mlmodel_decode];
        set_retained_object(ctx->model_decode, chunk_idx, model_decode);
        rwkv_coreml_firstchunk_implState *state = [model_decode newState];
        set_retained_object(ctx->states, chunk_idx, state);
        return true;
    }
    if (ctx->state_mode == RWKV_COREML_STATE_MODE_COREML) {
        rwkv_coreml_impl *model_decode = [[rwkv_coreml_impl alloc] initWithMLModel:mlmodel_decode];
        set_retained_object(ctx->model_decode, chunk_idx, model_decode);
        rwkv_coreml_implState *state = [model_decode newState];
        set_retained_object(ctx->states, chunk_idx, state);
        return true;
    }

    set_retained_object(ctx->model_decode, chunk_idx, mlmodel_decode);

    NSDictionary *model_inputs_decode = mlmodel_decode.modelDescription.inputDescriptionsByName;
    MLMultiArray *state_tokenshift = make_zero_multi_array_from_feature(model_inputs_decode, @"state_tokenshift_in");
    if (state_tokenshift == nil) {
        COREML_LOGE(@"Error getting state_tokenshift_in for state_mode %@", state_mode_name(ctx->state_mode));
        return false;
    }
    set_retained_object(ctx->state_tokenshift_tensors, chunk_idx, state_tokenshift);

    if (ctx->state_mode == RWKV_COREML_STATE_MODE_TENSOR) {
        MLMultiArray *state_wkv = make_zero_multi_array_from_feature(model_inputs_decode, @"state_wkv_in");
        if (state_wkv == nil) {
            COREML_LOGE(@"Error getting state_wkv_in for state_mode tensor");
            return false;
        }
        set_retained_object(ctx->state_wkv_tensors, chunk_idx, state_wkv);
    } else {
        MLState *state = [mlmodel_decode newState];
        if (state == nil) {
            COREML_LOGE(@"Error creating Core ML state for state_mode wkv-coreml");
            return false;
        }
        set_retained_object(ctx->states, chunk_idx, state);
    }
    return true;
}

static bool install_prefill_chunk(struct rwkv_coreml_context *ctx, MLModel *mlmodel_prefill, int chunk_idx) {
    if (ctx->state_mode == RWKV_COREML_STATE_MODE_COREML && ctx->num_chunks == 1) {
        rwkv_coreml_singlechunk_impl *model_prefill = [[rwkv_coreml_singlechunk_impl alloc] initWithMLModel:mlmodel_prefill];
        set_retained_object(ctx->model_prefill, chunk_idx, model_prefill);
        return true;
    }
    if (ctx->state_mode == RWKV_COREML_STATE_MODE_COREML && chunk_idx == 0) {
        rwkv_coreml_firstchunk_impl *model_prefill = [[rwkv_coreml_firstchunk_impl alloc] initWithMLModel:mlmodel_prefill];
        set_retained_object(ctx->model_prefill, chunk_idx, model_prefill);
        return true;
    }
    if (ctx->state_mode == RWKV_COREML_STATE_MODE_COREML) {
        rwkv_coreml_impl *model_prefill = [[rwkv_coreml_impl alloc] initWithMLModel:mlmodel_prefill];
        set_retained_object(ctx->model_prefill, chunk_idx, model_prefill);
        return true;
    }

    set_retained_object(ctx->model_prefill, chunk_idx, mlmodel_prefill);
    return true;
}

static bool read_prefill_seq_length(MLModel *mlmodel_prefill, int *prefill_seq_length_out) {
    NSDictionary *model_inputs_prefill = mlmodel_prefill.modelDescription.inputDescriptionsByName;
    NSArray<NSNumber *> *in_prefill_shape = get_shape_by_name(model_inputs_prefill, @"in0");
    if (in_prefill_shape == nil || in_prefill_shape.count < 2) {
        COREML_LOGE(@"Error getting in_prefill shape");
        return false;
    }
    if (prefill_seq_length_out) {
        *prefill_seq_length_out = [in_prefill_shape[1] intValue];
    }
    return true;
}

static MLMultiArray * external_state_wkv(struct rwkv_coreml_context *ctx, int chunk_idx) {
    return (__bridge MLMultiArray *)ctx->state_wkv_tensors[chunk_idx];
}

static MLMultiArray * external_state_tokenshift(struct rwkv_coreml_context *ctx, int chunk_idx) {
    return (__bridge MLMultiArray *)ctx->state_tokenshift_tensors[chunk_idx];
}

static bool update_external_state_from_output(struct rwkv_coreml_context *ctx, int chunk_idx, id<MLFeatureProvider> outFeatures) {
    MLFeatureValue *tokenshift_value = [outFeatures featureValueForName:@"state_tokenshift_out"];
    MLMultiArray *tokenshift_out = tokenshift_value.multiArrayValue;
    if (tokenshift_out == nil) {
        COREML_LOGE(@"Core ML output missing state_tokenshift_out");
        return false;
    }
    copy_multi_array(external_state_tokenshift(ctx, chunk_idx), tokenshift_out);

    if (ctx->state_mode == RWKV_COREML_STATE_MODE_TENSOR) {
        MLFeatureValue *wkv_value = [outFeatures featureValueForName:@"state_wkv_out"];
        MLMultiArray *wkv_out = wkv_value.multiArrayValue;
        if (wkv_out == nil) {
            COREML_LOGE(@"Core ML output missing state_wkv_out");
            return false;
        }
        copy_multi_array(external_state_wkv(ctx, chunk_idx), wkv_out);
    }
    return true;
}

static id<MLFeatureProvider> predict_generic_chunk(
    struct rwkv_coreml_context *ctx,
    bool prefill,
    int chunk_idx,
    MLMultiArray *in0,
    MLMultiArray *v_first_in
) {
    MLModel *model = (__bridge MLModel *)(prefill ? ctx->model_prefill[chunk_idx] : ctx->model_decode[chunk_idx]);
    NSMutableDictionary<NSString *, MLFeatureValue *> *features = [NSMutableDictionary dictionary];
    features[@"in0"] = multi_array_feature(in0);
    features[@"state_tokenshift_in"] = multi_array_feature(external_state_tokenshift(ctx, chunk_idx));
    if (ctx->state_mode == RWKV_COREML_STATE_MODE_TENSOR) {
        features[@"state_wkv_in"] = multi_array_feature(external_state_wkv(ctx, chunk_idx));
    }
    if (chunk_idx > 0 && ctx->num_chunks > 1) {
        if (v_first_in == nil) {
            COREML_LOGE(@"Core ML chunk %d missing v_first_in", chunk_idx);
            return nil;
        }
        features[@"v_first_in"] = multi_array_feature(v_first_in);
    }

    NSError *error = nil;
    MLDictionaryFeatureProvider *input = [[MLDictionaryFeatureProvider alloc] initWithDictionary:features error:&error];
    if (error || input == nil) {
        COREML_LOGE(@"Error creating Core ML feature provider for chunk %d: %@", chunk_idx, error);
        return nil;
    }

    MLPredictionOptions *options = [[MLPredictionOptions alloc] init];
    id<MLFeatureProvider> outFeatures = nil;
    if (ctx->state_mode == RWKV_COREML_STATE_MODE_WKV_COREML) {
        MLState *state = (__bridge MLState *)ctx->states[chunk_idx];
        outFeatures = [model predictionFromFeatures:input usingState:state options:options error:&error];
    } else {
        outFeatures = [model predictionFromFeatures:input options:options error:&error];
    }
    if (error || outFeatures == nil) {
        COREML_LOGE(@"Core ML prediction failed for chunk %d state_mode %@: %@", chunk_idx, state_mode_name(ctx->state_mode), error);
        return nil;
    }
    if (!update_external_state_from_output(ctx, chunk_idx, outFeatures)) {
        return nil;
    }
    return outFeatures;
}

static void* run_generic(struct rwkv_coreml_context *ctx, MLMultiArray *in0, bool prefill) {
    MLMultiArray *current = in0;
    MLMultiArray *v_first_out = nil;
    id<MLFeatureProvider> final_out = nil;

    for (int chunk_idx = 0; chunk_idx < ctx->num_chunks; ++chunk_idx) {
        id<MLFeatureProvider> outFeatures = predict_generic_chunk(ctx, prefill, chunk_idx, current, v_first_out);
        if (outFeatures == nil) return NULL;

        MLFeatureValue *out0_value = [outFeatures featureValueForName:@"out0"];
        current = out0_value.multiArrayValue;
        if (current == nil) {
            COREML_LOGE(@"Core ML output missing out0 for chunk %d", chunk_idx);
            return NULL;
        }

        if (chunk_idx == 0 && ctx->num_chunks > 1) {
            MLFeatureValue *v_first_value = [outFeatures featureValueForName:@"v_first_out"];
            v_first_out = v_first_value.multiArrayValue;
            if (v_first_out == nil) {
                COREML_LOGE(@"Core ML output missing v_first_out for chunk 0");
                return NULL;
            }
        }

        if (chunk_idx == ctx->num_chunks - 1) {
            final_out = outFeatures;
        }
    }

    if (prefill) {
        ctx->out_prefill = final_out;
    } else {
        ctx->out_decode = final_out;
    }
    return current.dataPointer;
}

static void with_state_wkv(struct rwkv_coreml_context *ctx, int chunk_idx, void (^handler)(MLMultiArray *buffer)) {
    if (ctx->state_mode == RWKV_COREML_STATE_MODE_TENSOR) {
        MLMultiArray *buffer = (__bridge MLMultiArray *)ctx->state_wkv_tensors[chunk_idx];
        handler(buffer);
        return;
    }
    if (ctx->state_mode == RWKV_COREML_STATE_MODE_WKV_COREML) {
        MLState *state = (__bridge MLState *)ctx->states[chunk_idx];
        [state getMultiArrayForStateNamed:@"state_wkv" handler:handler];
        return;
    }
    if (ctx->num_chunks == 1) {
        rwkv_coreml_singlechunk_implState *state = (__bridge rwkv_coreml_singlechunk_implState *)ctx->states[0];
        [state getMultiArrayForState:rwkv_coreml_singlechunk_implStateNameState_wkv handler:handler];
        return;
    }
    if (chunk_idx == 0) {
        rwkv_coreml_firstchunk_implState *state = (__bridge rwkv_coreml_firstchunk_implState *)ctx->states[chunk_idx];
        [state getMultiArrayForState:rwkv_coreml_firstchunk_implStateNameState_wkv handler:handler];
        return;
    }
    rwkv_coreml_implState *state = (__bridge rwkv_coreml_implState *)ctx->states[chunk_idx];
    [state getMultiArrayForState:rwkv_coreml_implStateNameState_wkv handler:handler];
}

static void with_state_tokenshift(struct rwkv_coreml_context *ctx, int chunk_idx, void (^handler)(MLMultiArray *buffer)) {
    if (ctx->state_mode == RWKV_COREML_STATE_MODE_TENSOR || ctx->state_mode == RWKV_COREML_STATE_MODE_WKV_COREML) {
        MLMultiArray *buffer = (__bridge MLMultiArray *)ctx->state_tokenshift_tensors[chunk_idx];
        handler(buffer);
        return;
    }
    if (ctx->num_chunks == 1) {
        rwkv_coreml_singlechunk_implState *state = (__bridge rwkv_coreml_singlechunk_implState *)ctx->states[0];
        [state getMultiArrayForState:rwkv_coreml_singlechunk_implStateNameState_tokenshift handler:handler];
        return;
    }
    if (chunk_idx == 0) {
        rwkv_coreml_firstchunk_implState *state = (__bridge rwkv_coreml_firstchunk_implState *)ctx->states[chunk_idx];
        [state getMultiArrayForState:rwkv_coreml_firstchunk_implStateNameState_tokenshift handler:handler];
        return;
    }
    rwkv_coreml_implState *state = (__bridge rwkv_coreml_implState *)ctx->states[chunk_idx];
    [state getMultiArrayForState:rwkv_coreml_implStateNameState_tokenshift handler:handler];
}

struct rwkv_coreml_context * rwkv_coreml_new_context(void) {
    return new rwkv_coreml_context;
}

int rwkv_coreml_init(struct rwkv_coreml_context * ctx, const char * path_model, int load_prefill_async, int async_prefill_decode_load_threshold_ms) {
    @autoreleasepool {
        if (!ctx || !path_model) {
            return -1;
        }
        rwkv_coreml_release_resources(ctx);
        NSString * path_model_str = [[NSString alloc] initWithUTF8String:path_model];

        NSString *config_path = [path_model_str stringByAppendingPathComponent:@"config.yaml"];
        NSString *basename = nil;
        int num_chunks = 0;
        rwkv_coreml_state_mode state_mode = RWKV_COREML_STATE_MODE_COREML;
        if (!parse_coreml_config(config_path, &basename, &num_chunks, &state_mode)) {
            return -1;
        }
        ctx->state_mode = state_mode;
        const bool requested_async_prefill = load_prefill_async != 0;
        COREML_LOGI(@"Initializing RWKV CoreML with model at %@, basename=%@, num_chunks=%d, state_mode=%@, requested_async_prefill=%d",
              path_model_str, basename, num_chunks, state_mode_name(state_mode), requested_async_prefill ? 1 : 0);
        const int async_prefill_threshold_ms = async_prefill_decode_load_threshold_ms == 0
            ? kDefaultAsyncPrefillDecodeLoadThresholdMs
            : async_prefill_decode_load_threshold_ms;
        bool async_prefill = false;
        ctx->load_prefill_async.store(false);

        // select which device to run the Core ML model on
        MLModelConfiguration *config_decode = [[MLModelConfiguration alloc] init];
        config_decode.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
        config_decode.functionName = @"decode";

        MLModelConfiguration *config_prefill = [[MLModelConfiguration alloc] init];
        config_prefill.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
        config_prefill.functionName = @"prefill";

        ctx->num_chunks = num_chunks;
        ctx->model_decode.resize((size_t)num_chunks, nullptr);
        ctx->model_prefill.resize((size_t)num_chunks, nullptr);
        ctx->states.resize((size_t)num_chunks, nullptr);
        ctx->state_wkv_tensors.resize((size_t)num_chunks, nullptr);
        ctx->state_tokenshift_tensors.resize((size_t)num_chunks, nullptr);
        ctx->state_wkv_bytes_per_chunk.resize((size_t)num_chunks, 0);
        ctx->state_tokenshift_bytes_per_chunk.resize((size_t)num_chunks, 0);

        int total_layers = 0;
        int num_heads = 0;
        int head_dim = 0;
        int prefill_seq_length = 0;
        int vocab_size = 0;

        auto total_start = std::chrono::steady_clock::now();
        ctx->load_done_chunks.store(0);
        ctx->load_progress_reported = 0.f;
        ctx->prefill_ready.store(false);
        ctx->prefill_failed.store(false);
        for (int chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
            NSString *model_name = nil;
            model_name = [NSString stringWithFormat:@"%@_chunk%dof%d.mlmodelc", basename, chunk_idx + 1, num_chunks];
            NSString *model_path = [path_model_str stringByAppendingPathComponent:model_name];
            NSURL *url_model = [NSURL fileURLWithPath:model_path];

            MLModel *mlmodel_decode = load_coreml_model_with_retry(url_model, config_decode, model_name, @"decode", chunk_idx, num_chunks);
            if (!mlmodel_decode) {
                COREML_LOGE(@"Error loading decode model %@ after retries", model_name);
                rwkv_coreml_release_resources(ctx);
                return -1;
            }
            if (!install_decode_chunk(ctx, mlmodel_decode, chunk_idx)) {
                rwkv_coreml_release_resources(ctx);
                return -1;
            }
            ctx->load_done_chunks.store(requested_async_prefill ? chunk_idx + 1 : chunk_idx * 2 + 1);

            if (!requested_async_prefill) {
                MLModel *mlmodel_prefill = load_coreml_model_with_retry(url_model, config_prefill, model_name, @"prefill", chunk_idx, num_chunks);
                if (!mlmodel_prefill) {
                    COREML_LOGE(@"Error loading prefill model %@ after retries", model_name);
                    rwkv_coreml_release_resources(ctx);
                    return -1;
                }
                if (!install_prefill_chunk(ctx, mlmodel_prefill, chunk_idx)) {
                    rwkv_coreml_release_resources(ctx);
                    return -1;
                }
                if (chunk_idx == 0 && !read_prefill_seq_length(mlmodel_prefill, &prefill_seq_length)) {
                    rwkv_coreml_release_resources(ctx);
                    return -1;
                }
                ctx->load_done_chunks.store(chunk_idx * 2 + 2);
            }

            NSDictionary *model_outputs = mlmodel_decode.modelDescription.outputDescriptionsByName;
            if (chunk_idx == num_chunks - 1) {
                NSArray<NSNumber *> *logits_out_shape = get_shape_by_name(model_outputs, @"out0");
                if (logits_out_shape == nil) {
                    COREML_LOGE(@"Error getting out0 shape");
                    rwkv_coreml_release_resources(ctx);
                    return -1;
                }
                vocab_size = [logits_out_shape[2] intValue];
            }

            __block MLMultiArray *state_wkv = nil;
            with_state_wkv(ctx, chunk_idx, ^(MLMultiArray *buffer) {
                state_wkv = buffer;
            });
            NSArray<NSNumber *> *state_wkv_shape = state_wkv.shape;
            if (state_wkv_shape == nil) {
                COREML_LOGE(@"Error getting state_wkv shape");
                rwkv_coreml_release_resources(ctx);
                return -1;
            }
            total_layers += [state_wkv_shape[0] intValue];
            int chunk_num_heads = [state_wkv_shape[1] intValue];
            int chunk_head_dim = [state_wkv_shape[2] intValue];
            if (num_heads == 0) num_heads = chunk_num_heads;
            if (head_dim == 0) head_dim = chunk_head_dim;
            if (num_heads != chunk_num_heads || head_dim != chunk_head_dim) {
                COREML_LOGW(@"Warning: inconsistent head shape in chunk %d (heads=%d dim=%d)", chunk_idx, chunk_num_heads, chunk_head_dim);
            }

            // Cache exact byte sizes for state buffers (do NOT assume shapes).
            with_state_wkv(ctx, chunk_idx, ^(MLMultiArray *buffer) {
                [buffer getBytesWithHandler:^(const void *bytes, NSInteger size) {
                    (void)bytes;
                    ctx->state_wkv_bytes_per_chunk[chunk_idx] = (size_t)size;
                }];
            });
            with_state_tokenshift(ctx, chunk_idx, ^(MLMultiArray *buffer) {
                [buffer getBytesWithHandler:^(const void *bytes, NSInteger size) {
                    (void)bytes;
                    ctx->state_tokenshift_bytes_per_chunk[chunk_idx] = (size_t)size;
                }];
            });
        }

        auto decode_end = std::chrono::steady_clock::now();
        double decode_load_ms = std::chrono::duration<double, std::milli>(decode_end - total_start).count();
        if (requested_async_prefill) {
            async_prefill = async_prefill_threshold_ms < 0 || decode_load_ms >= async_prefill_threshold_ms;
            ctx->load_prefill_async.store(async_prefill);
            if (async_prefill) {
                if (async_prefill_threshold_ms < 0) {
                    COREML_LOGI(@"Decode model load time: %.2f ms (async prefill forced; threshold disabled)",
                          decode_load_ms);
                } else {
                    COREML_LOGI(@"Decode model load time: %.2f ms >= %d ms threshold (prefill loading in background)",
                          decode_load_ms, async_prefill_threshold_ms);
                }
            } else {
                COREML_LOGI(@"Decode model load time: %.2f ms < %d ms threshold (loading prefill synchronously)",
                      decode_load_ms, async_prefill_threshold_ms);
                auto prefill_sync_start = std::chrono::steady_clock::now();
                for (int chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
                    NSString *model_name = [NSString stringWithFormat:@"%@_chunk%dof%d.mlmodelc", basename, chunk_idx + 1, num_chunks];
                    NSString *model_path = [path_model_str stringByAppendingPathComponent:model_name];
                    NSURL *url_model = [NSURL fileURLWithPath:model_path];
                    MLModel *mlmodel_prefill = load_coreml_model_with_retry(url_model, config_prefill, model_name, @"prefill", chunk_idx, num_chunks);
                    if (!mlmodel_prefill) {
                        COREML_LOGE(@"Error loading prefill model %@ after retries", model_name);
                        rwkv_coreml_release_resources(ctx);
                        return -1;
                    }
                    if (!install_prefill_chunk(ctx, mlmodel_prefill, chunk_idx)) {
                        rwkv_coreml_release_resources(ctx);
                        return -1;
                    }
                    if (chunk_idx == 0 && !read_prefill_seq_length(mlmodel_prefill, &prefill_seq_length)) {
                        rwkv_coreml_release_resources(ctx);
                        return -1;
                    }
                    ctx->load_done_chunks.store(num_chunks + chunk_idx + 1);
                }
                auto prefill_sync_end = std::chrono::steady_clock::now();
                double prefill_sync_ms = std::chrono::duration<double, std::milli>(prefill_sync_end - prefill_sync_start).count();
                COREML_LOGI(@"CoreML async prefill skipped by threshold; synchronous prefill load time: %.2f ms",
                      prefill_sync_ms);
            }
        }

        if (async_prefill && prefill_seq_length <= 0) {
            prefill_seq_length = 1;
        }
        ctx->prefill_seq_length.store(std::max(1, prefill_seq_length));
        ctx->n_layers = total_layers;
        ctx->num_heads = num_heads;
        ctx->head_dim = head_dim;
        ctx->embd_dim = ctx->head_dim * ctx->num_heads;
        ctx->vocab_size = vocab_size;

        auto total_end = std::chrono::steady_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();
        if (!requested_async_prefill) {
            COREML_LOGI(@"Total model load time: %.2f ms", total_ms);
        } else if (!async_prefill) {
            COREML_LOGI(@"Total model load time: %.2f ms (async prefill skipped)", total_ms);
        }

        ctx->state_wkv_bytes = 0;
        ctx->state_tokenshift_bytes = 0;
        for (int i = 0; i < num_chunks; ++i) {
            ctx->state_wkv_bytes += ctx->state_wkv_bytes_per_chunk[i];
            ctx->state_tokenshift_bytes += ctx->state_tokenshift_bytes_per_chunk[i];
        }

        if (async_prefill) {
            const std::string model_dir(path_model);
            const std::string basename_cstr([basename UTF8String]);
            ctx->prefill_load_thread = std::thread([ctx, model_dir, basename_cstr, num_chunks]() {
                @autoreleasepool {
                    MLModelConfiguration *config_prefill_bg = [[MLModelConfiguration alloc] init];
                    config_prefill_bg.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
                    config_prefill_bg.functionName = @"prefill";
                    NSString *path_model_str_bg = [[NSString alloc] initWithUTF8String:model_dir.c_str()];
                    NSString *basename_bg = [[NSString alloc] initWithUTF8String:basename_cstr.c_str()];
                    int loaded_prefill_seq_length = 0;
                    auto prefill_start = std::chrono::steady_clock::now();
                    for (int chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
                        NSString *model_name = [NSString stringWithFormat:@"%@_chunk%dof%d.mlmodelc", basename_bg, chunk_idx + 1, num_chunks];
                        NSString *model_path = [path_model_str_bg stringByAppendingPathComponent:model_name];
                        NSURL *url_model = [NSURL fileURLWithPath:model_path];
                        MLModel *mlmodel_prefill = load_coreml_model_with_retry(url_model, config_prefill_bg, model_name, @"prefill", chunk_idx, num_chunks);
                        if (!mlmodel_prefill) {
                            COREML_LOGE(@"Error loading prefill model %@ after retries in background", model_name);
                            ctx->prefill_failed.store(true);
                            COREML_LOGE(@"CoreML async prefill load failed; prompt eval will continue using decode");
                            return;
                        }
                        if (!install_prefill_chunk(ctx, mlmodel_prefill, chunk_idx)) {
                            ctx->prefill_failed.store(true);
                            COREML_LOGE(@"CoreML async prefill install failed; prompt eval will continue using decode");
                            return;
                        }
                        if (chunk_idx == 0 && !read_prefill_seq_length(mlmodel_prefill, &loaded_prefill_seq_length)) {
                            ctx->prefill_failed.store(true);
                            COREML_LOGE(@"CoreML async prefill shape read failed; prompt eval will continue using decode");
                            return;
                        }
                        ctx->load_done_chunks.store(num_chunks + chunk_idx + 1);
                    }
                    if (loaded_prefill_seq_length > 0) {
                        ctx->prefill_seq_length.store(loaded_prefill_seq_length);
                    }
                    ctx->prefill_ready.store(true, std::memory_order_release);
                    auto prefill_end = std::chrono::steady_clock::now();
                    double prefill_ms = std::chrono::duration<double, std::milli>(prefill_end - prefill_start).count();
                    COREML_LOGI(@"CoreML async prefill ready; switching future prompt eval to prefill. load_time: %.2f ms, prefill_seq_length: %d",
                          prefill_ms, ctx->prefill_seq_length.load());
                }
            });
        } else {
            ctx->prefill_ready.store(true, std::memory_order_release);
        }

        COREML_LOGI(@"state_mode: %@, num_chunks: %d, num_heads: %d, head_dim: %d, vocab_size: %d, n_layers: %d, prefill_seq_length: %d, state_wkv_bytes: %zu, state_tokenshift_bytes: %zu\n",
            state_mode_name(ctx->state_mode), ctx->num_chunks, ctx->num_heads, ctx->head_dim, ctx->vocab_size, ctx->n_layers, ctx->prefill_seq_length.load(), ctx->state_wkv_bytes, ctx->state_tokenshift_bytes);
        return 0;
    }
}

void rwkv_coreml_free(struct rwkv_coreml_context * ctx) {
    if (ctx) {
        rwkv_coreml_release_resources(ctx);
        delete ctx;
    }
}

float rwkv_coreml_get_load_progress(struct rwkv_coreml_context * ctx) {
    if (!ctx || ctx->num_chunks <= 0) return 1.0f;
    int total_steps = std::max(1, ctx->load_prefill_async.load() ? ctx->num_chunks : ctx->num_chunks * 2);
    int done_steps = std::max(0, std::min(total_steps, ctx->load_done_chunks.load()));
    float real = (float)done_steps / (float)total_steps;
    float ceiling = (done_steps + 1 <= total_steps)
        ? (float)(done_steps + 1) / (float)total_steps
        : 1.f;
    if (ctx->load_progress_reported < real) {
        ctx->load_progress_reported = real;
    }

    const float min_step = 0.0005f;
    float ret = ctx->load_progress_reported;
    float remaining = std::max(1e-5f, ceiling - ctx->load_progress_reported);
    float step = remaining * 0.01f;
    step = std::max(min_step, step);
    ctx->load_progress_reported = std::min(ceiling - 0.01f, ctx->load_progress_reported + step);
    return std::max(0.f, std::min(1.f, ctx->load_progress_reported));
}

int rwkv_coreml_is_prefill_ready(struct rwkv_coreml_context * ctx) {
    if (!ctx) return 0;
    return ctx->prefill_ready.load(std::memory_order_acquire) ? 1 : 0;
}

void* rwkv_coreml_decode(struct rwkv_coreml_context * ctx, int token) {
    // Ensure temporary autoreleased CoreML objects don't accumulate in tight loops,
    // while keeping the last output alive (ctx->out_decode is __strong).
    @autoreleasepool {
        MLMultiArray * inMultiArray = [
            [MLMultiArray alloc] initWithDataPointer: &token
                                               shape: @[@1, @(1)]
                                            dataType: MLMultiArrayDataTypeInt32
                                             strides: @[@(1), @(1)]
                                         deallocator: nil
                                               error: nil
        ];

        if (ctx->state_mode != RWKV_COREML_STATE_MODE_COREML) {
            return run_generic(ctx, inMultiArray, false);
        }

        if (ctx->num_chunks == 1) {
            rwkv_coreml_singlechunk_impl *model_decode = (__bridge rwkv_coreml_singlechunk_impl *)ctx->model_decode[0];
            rwkv_coreml_singlechunk_implState *state = (__bridge rwkv_coreml_singlechunk_implState *)ctx->states[0];
            ctx->out_decode = [model_decode predictionFromIn0: inMultiArray usingState: state error: nil];
            return [(rwkv_coreml_singlechunk_implOutput *)ctx->out_decode out0].dataPointer;
        }

        rwkv_coreml_firstchunk_impl *first_model_decode = (__bridge rwkv_coreml_firstchunk_impl *)ctx->model_decode[0];
        rwkv_coreml_firstchunk_implState *first_state = (__bridge rwkv_coreml_firstchunk_implState *)ctx->states[0];
        rwkv_coreml_firstchunk_implOutput *first_out = [first_model_decode predictionFromIn0: inMultiArray usingState: first_state error: nil];
        MLMultiArray *current = first_out.out0;
        MLMultiArray *v_first_out = first_out.v_first_out;

        for (int chunk_idx = 1; chunk_idx < ctx->num_chunks; ++chunk_idx) {
            rwkv_coreml_impl *model_decode = (__bridge rwkv_coreml_impl *)ctx->model_decode[chunk_idx];
            rwkv_coreml_implState *state = (__bridge rwkv_coreml_implState *)ctx->states[chunk_idx];
            rwkv_coreml_implOutput *out = [model_decode predictionFromIn0: current v_first_in: v_first_out usingState: state error: nil];
            current = out.out0;
            if (chunk_idx == ctx->num_chunks - 1) {
                ctx->out_decode = out;
            }
        }
        return [(rwkv_coreml_implOutput *)ctx->out_decode out0].dataPointer;
    }
}

void* rwkv_coreml_prefill(struct rwkv_coreml_context * ctx, std::vector<int> tokens) {
    // See rwkv_coreml_decode() for why we use autoreleasepool here.
    @autoreleasepool {
        if (!ctx->prefill_ready.load(std::memory_order_acquire)) {
            COREML_LOGE(@"Core ML prefill requested before prefill function is ready");
            return NULL;
        }
        int prefill_seq_length = ctx->prefill_seq_length.load();
        if (tokens.size() != (size_t)prefill_seq_length) {
            COREML_LOGE(@"Error: tokens size is not equal to prefill_seq_length");
            return NULL;
        }
        MLMultiArray * inMultiArray = [
            [MLMultiArray alloc] initWithDataPointer: tokens.data()
                                               shape: @[@1, @(tokens.size())]
                                            dataType: MLMultiArrayDataTypeInt32
                                             strides: @[@(tokens.size()), @(1)]
                                         deallocator: nil
                                               error: nil
        ];

        if (ctx->state_mode != RWKV_COREML_STATE_MODE_COREML) {
            return run_generic(ctx, inMultiArray, true);
        }

        if (ctx->num_chunks == 1) {
            rwkv_coreml_singlechunk_impl *model_prefill = (__bridge rwkv_coreml_singlechunk_impl *)ctx->model_prefill[0];
            rwkv_coreml_singlechunk_implState *state = (__bridge rwkv_coreml_singlechunk_implState *)ctx->states[0];
            ctx->out_prefill = [model_prefill predictionFromIn0: inMultiArray usingState: state error: nil];
            return [(rwkv_coreml_singlechunk_implOutput *)ctx->out_prefill out0].dataPointer;
        }

        rwkv_coreml_firstchunk_impl *first_model_prefill = (__bridge rwkv_coreml_firstchunk_impl *)ctx->model_prefill[0];
        rwkv_coreml_firstchunk_implState *first_state = (__bridge rwkv_coreml_firstchunk_implState *)ctx->states[0];
        rwkv_coreml_firstchunk_implOutput *first_out = [first_model_prefill predictionFromIn0: inMultiArray usingState: first_state error: nil];
        MLMultiArray *current = first_out.out0;
        MLMultiArray *v_first_out = first_out.v_first_out;

        for (int chunk_idx = 1; chunk_idx < ctx->num_chunks; ++chunk_idx) {
            rwkv_coreml_impl *model_prefill = (__bridge rwkv_coreml_impl *)ctx->model_prefill[chunk_idx];
            rwkv_coreml_implState *state = (__bridge rwkv_coreml_implState *)ctx->states[chunk_idx];
            rwkv_coreml_implOutput *out = [model_prefill predictionFromIn0: current v_first_in: v_first_out usingState: state error: nil];
            current = out.out0;
            if (chunk_idx == ctx->num_chunks - 1) {
                ctx->out_prefill = out;
            }
        }
        return [(rwkv_coreml_implOutput *)ctx->out_prefill out0].dataPointer;
    }
}

int rwkv_coreml_get_vocab_size(struct rwkv_coreml_context * ctx) {
    return ctx->vocab_size;
}

int rwkv_coreml_get_n_layers(struct rwkv_coreml_context * ctx) {
    return ctx->n_layers;
}

int rwkv_coreml_get_num_heads(struct rwkv_coreml_context * ctx) {
    return ctx->num_heads;
}

int rwkv_coreml_get_head_dim(struct rwkv_coreml_context * ctx) {
    return ctx->head_dim;
}

int rwkv_coreml_get_hidden_dim(struct rwkv_coreml_context * ctx) {
    return ctx->embd_dim;
}

int rwkv_coreml_get_prefill_seq_length(struct rwkv_coreml_context * ctx) {
    return ctx->prefill_seq_length.load();
}

int rwkv_coreml_get_state_wkv_bytes(struct rwkv_coreml_context * ctx) {
    return ctx->state_wkv_bytes;
}

int rwkv_coreml_get_state_tokenshift_bytes(struct rwkv_coreml_context * ctx) {
    return ctx->state_tokenshift_bytes;
}

std::vector<std::vector<uint8_t>> rwkv_coreml_get_state(struct rwkv_coreml_context * ctx) {
    std::vector<std::vector<uint8_t>> state_ret(2); // wkv and tokenshift
    if (!ctx || ctx->num_chunks <= 0) {
        COREML_LOGE(@"rwkv_coreml_get_state: invalid ctx/state");
        return state_ret;
    }
    if (ctx->state_wkv_bytes > 0) state_ret[0].resize(ctx->state_wkv_bytes);
    if (ctx->state_tokenshift_bytes > 0) state_ret[1].resize(ctx->state_tokenshift_bytes);
    uint8_t * wkv_dst = state_ret[0].empty() ? nullptr : state_ret[0].data();
    const size_t wkv_dst_size = state_ret[0].size();
    uint8_t * tokenshift_dst = state_ret[1].empty() ? nullptr : state_ret[1].data();
    const size_t tokenshift_dst_size = state_ret[1].size();

    size_t wkv_offset = 0;
    size_t tokenshift_offset = 0;
    for (int chunk_idx = 0; chunk_idx < ctx->num_chunks; ++chunk_idx) {
        const size_t wkv_bytes = ctx->state_wkv_bytes_per_chunk[chunk_idx];
        const size_t tokenshift_bytes = ctx->state_tokenshift_bytes_per_chunk[chunk_idx];
        with_state_wkv(ctx, chunk_idx, ^(MLMultiArray *buffer) {
            [buffer getBytesWithHandler:^(const void *bytes, NSInteger size) {
                if (bytes == nullptr || size <= 0) return;
                const size_t src_size = (size_t)size;
                if (wkv_dst == nullptr || wkv_dst_size == 0) {
                    COREML_LOGE(@"rwkv_coreml_get_state: state_wkv dst buffer is empty (init-time size not captured?) src=%zu", src_size);
                    return;
                }
                if (src_size != wkv_bytes) {
                    COREML_LOGW(@"rwkv_coreml_get_state: state_wkv size mismatch: src=%zu expected=%zu", src_size, wkv_bytes);
                }
                const size_t dst_remaining = wkv_dst_size - wkv_offset;
                const size_t n = std::min(dst_remaining, std::min(wkv_bytes, src_size));
                if (n > 0) std::memcpy(wkv_dst + wkv_offset, bytes, n);
            }];
        });
        with_state_tokenshift(ctx, chunk_idx, ^(MLMultiArray *buffer) {
            [buffer getBytesWithHandler:^(const void *bytes, NSInteger size) {
                if (bytes == nullptr || size <= 0) return;
                const size_t src_size = (size_t)size;
                if (tokenshift_dst == nullptr || tokenshift_dst_size == 0) {
                    COREML_LOGE(@"rwkv_coreml_get_state: state_tokenshift dst buffer is empty (init-time size not captured?) src=%zu", src_size);
                    return;
                }
                if (src_size != tokenshift_bytes) {
                    COREML_LOGW(@"rwkv_coreml_get_state: state_tokenshift size mismatch: src=%zu expected=%zu", src_size, tokenshift_bytes);
                }
                const size_t dst_remaining = tokenshift_dst_size - tokenshift_offset;
                const size_t n = std::min(dst_remaining, std::min(tokenshift_bytes, src_size));
                if (n > 0) std::memcpy(tokenshift_dst + tokenshift_offset, bytes, n);
            }];
        });
        wkv_offset += wkv_bytes;
        tokenshift_offset += tokenshift_bytes;
    }
    return state_ret;
}

void rwkv_coreml_set_state(struct rwkv_coreml_context * ctx, std::vector<std::vector<uint8_t>> state) {
    if (!ctx || ctx->num_chunks <= 0) {
        COREML_LOGE(@"rwkv_coreml_set_state: invalid ctx/state");
        return;
    }
    if (state.size() < 2) {
        COREML_LOGE(@"rwkv_coreml_set_state: invalid state vector size: %zu", state.size());
        return;
    }
    size_t wkv_offset = 0;
    size_t tokenshift_offset = 0;
    for (int chunk_idx = 0; chunk_idx < ctx->num_chunks; ++chunk_idx) {
        const size_t wkv_bytes = ctx->state_wkv_bytes_per_chunk[chunk_idx];
        const size_t tokenshift_bytes = ctx->state_tokenshift_bytes_per_chunk[chunk_idx];
        with_state_wkv(ctx, chunk_idx, ^(MLMultiArray *buffer) {
            [buffer getMutableBytesWithHandler:^(void *mutableBytes, NSInteger size, NSArray<NSNumber *> *strides) {
                (void)strides;
                if (mutableBytes == nullptr || size <= 0) return;
                const size_t dst_size = (size_t)size;
                const size_t src_size = state[0].size();
                if (dst_size != wkv_bytes) {
                    COREML_LOGW(@"rwkv_coreml_set_state: state_wkv size mismatch: dst=%zu expected=%zu", dst_size, wkv_bytes);
                }
                const size_t src_remaining = src_size > wkv_offset ? src_size - wkv_offset : 0;
                const size_t n = std::min(dst_size, std::min(wkv_bytes, src_remaining));
                if (n > 0) std::memcpy(mutableBytes, state[0].data() + wkv_offset, n);
                if (dst_size > n) std::memset((uint8_t*)mutableBytes + n, 0, dst_size - n);
            }];
        });
        with_state_tokenshift(ctx, chunk_idx, ^(MLMultiArray *buffer) {
            [buffer getMutableBytesWithHandler:^(void *mutableBytes, NSInteger size, NSArray<NSNumber *> *strides) {
                (void)strides;
                if (mutableBytes == nullptr || size <= 0) return;
                const size_t dst_size = (size_t)size;
                const size_t src_size = state[1].size();
                if (dst_size != tokenshift_bytes) {
                    COREML_LOGW(@"rwkv_coreml_set_state: state_tokenshift size mismatch: dst=%zu expected=%zu", dst_size, tokenshift_bytes);
                }
                const size_t src_remaining = src_size > tokenshift_offset ? src_size - tokenshift_offset : 0;
                const size_t n = std::min(dst_size, std::min(tokenshift_bytes, src_remaining));
                if (n > 0) std::memcpy(mutableBytes, state[1].data() + tokenshift_offset, n);
                if (dst_size > n) std::memset((uint8_t*)mutableBytes + n, 0, dst_size - n);
            }];
        });
        wkv_offset += wkv_bytes;
        tokenshift_offset += tokenshift_bytes;
    }
}

void rwkv_coreml_set_wkv_state(struct rwkv_coreml_context * ctx, std::vector<half_float::half> state) {
    if (!ctx || ctx->num_chunks <= 0) {
        COREML_LOGE(@"rwkv_coreml_set_wkv_state: invalid ctx/state");
        return;
    }
    if (state.size() * sizeof(half_float::half) != ctx->state_wkv_bytes) {
        COREML_LOGE(@"rwkv_coreml_set_wkv_state: invalid state vector size: %zu", state.size() * sizeof(half_float::half));
        return;
    }
    uint8_t *src = (uint8_t *)state.data();
    size_t offset = 0;
    for (int chunk_idx = 0; chunk_idx < ctx->num_chunks; ++chunk_idx) {
        const size_t wkv_bytes = ctx->state_wkv_bytes_per_chunk[chunk_idx];
        with_state_wkv(ctx, chunk_idx, ^(MLMultiArray *buffer) {
            [buffer getMutableBytesWithHandler:^(void *mutableBytes, NSInteger size, NSArray<NSNumber *> *strides) {
                (void)strides;
                const size_t dst_size = (size_t)size;
                const size_t n = std::min(dst_size, wkv_bytes);
                if (n > 0) std::memcpy(mutableBytes, src + offset, n);
                if (dst_size > n) std::memset((uint8_t*)mutableBytes + n, 0, dst_size - n);
            }];
        });
        offset += wkv_bytes;
    }
}

void rwkv_coreml_zero_state(struct rwkv_coreml_context * ctx) {
    if (!ctx || ctx->num_chunks <= 0) return;
    for (int chunk_idx = 0; chunk_idx < ctx->num_chunks; ++chunk_idx) {
        with_state_wkv(ctx, chunk_idx, ^(MLMultiArray *buffer) {
            [buffer getMutableBytesWithHandler:^(void *mutableBytes, NSInteger size, NSArray<NSNumber *> *strides) {
                (void)strides;
                std::memset((void*)mutableBytes, 0, (size_t)size);
            }];
        });
        with_state_tokenshift(ctx, chunk_idx, ^(MLMultiArray *buffer) {
            [buffer getMutableBytesWithHandler:^(void *mutableBytes, NSInteger size, NSArray<NSNumber *> *strides) {
                (void)strides;
                std::memset((void*)mutableBytes, 0, (size_t)size);
            }];
        });
    }
}
