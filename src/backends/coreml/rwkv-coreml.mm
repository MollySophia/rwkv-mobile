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
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <vector>
#include <chrono>
#include "half.hpp"
#include "logger.h"

struct rwkv_coreml_context {
    std::vector<const void *> model_decode;
    std::vector<const void *> model_prefill;
    std::vector<const void *> states;
    int num_chunks = 0;
    int load_done_chunks = 0;
    float load_progress_reported = 0.f;
    int n_layers;
    int num_heads;
    int head_dim;
    int embd_dim;
    int vocab_size;
    int prefill_seq_length;

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

static void rwkv_coreml_release_resources(struct rwkv_coreml_context * ctx) {
    if (!ctx) return;
    for (size_t i = 0; i < ctx->model_decode.size(); ++i) {
        if (ctx->model_decode[i]) CFRelease(ctx->model_decode[i]);
    }
    for (size_t i = 0; i < ctx->model_prefill.size(); ++i) {
        if (ctx->model_prefill[i]) CFRelease(ctx->model_prefill[i]);
    }
    for (size_t i = 0; i < ctx->states.size(); ++i) {
        if (ctx->states[i]) CFRelease(ctx->states[i]);
    }
    ctx->model_decode.clear();
    ctx->model_prefill.clear();
    ctx->states.clear();
    ctx->state_wkv_bytes_per_chunk.clear();
    ctx->state_tokenshift_bytes_per_chunk.clear();
    ctx->state_wkv_bytes = 0;
    ctx->state_tokenshift_bytes = 0;
    ctx->num_chunks = 0;
    ctx->load_done_chunks = 0;
    ctx->load_progress_reported = 0.f;
    ctx->n_layers = 0;
    ctx->num_heads = 0;
    ctx->head_dim = 0;
    ctx->embd_dim = 0;
    ctx->vocab_size = 0;
    ctx->prefill_seq_length = 0;
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

static bool parse_coreml_config(NSString *config_path, NSString **basename_out, int *num_chunks_out) {
    NSError *error = nil;
    NSString *content = [NSString stringWithContentsOfFile:config_path encoding:NSUTF8StringEncoding error:&error];
    if (error || content == nil) {
        NSLog(@"Error reading config.yaml: %@", error);
        return false;
    }
    NSString *basename = nil;
    int num_chunks = 0;
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
        }
    }
    if (basename == nil || basename.length == 0) {
        NSLog(@"config.yaml missing basename");
        return false;
    }
    if (num_chunks <= 0) {
        NSLog(@"config.yaml invalid num_chunks: %d", num_chunks);
        return false;
    }
    if (basename_out) *basename_out = basename;
    if (num_chunks_out) *num_chunks_out = num_chunks;
    return true;
}

static void with_state_wkv(struct rwkv_coreml_context *ctx, int chunk_idx, void (^handler)(MLMultiArray *buffer)) {
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

int rwkv_coreml_init(struct rwkv_coreml_context * ctx, const char * path_model) {
    @autoreleasepool {
        if (!ctx || !path_model) {
            return -1;
        }
        rwkv_coreml_release_resources(ctx);
        NSString * path_model_str = [[NSString alloc] initWithUTF8String:path_model];

        NSString *config_path = [path_model_str stringByAppendingPathComponent:@"config.yaml"];
        NSString *basename = nil;
        int num_chunks = 0;
        if (!parse_coreml_config(config_path, &basename, &num_chunks)) {
            return -1;
        }

        // select which device to run the Core ML model on
        MLModelConfiguration *config_decode = [[MLModelConfiguration alloc] init];
        config_decode.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
        config_decode.functionName = @"decode";

        MLModelConfiguration *config_prefill = [[MLModelConfiguration alloc] init];
        config_prefill.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
        config_prefill.functionName = @"prefill";

        NSError *error = nil;

        ctx->num_chunks = num_chunks;
        ctx->model_decode.reserve((size_t)num_chunks);
        ctx->model_prefill.reserve((size_t)num_chunks);
        ctx->states.reserve((size_t)num_chunks);
        ctx->state_wkv_bytes_per_chunk.resize((size_t)num_chunks, 0);
        ctx->state_tokenshift_bytes_per_chunk.resize((size_t)num_chunks, 0);

        int total_layers = 0;
        int num_heads = 0;
        int head_dim = 0;
        int prefill_seq_length = 0;
        int vocab_size = 0;

        auto total_start = std::chrono::steady_clock::now();
        ctx->load_done_chunks = 0;
        ctx->load_progress_reported = 0.f;
        for (int chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
            NSString *model_name = nil;
            model_name = [NSString stringWithFormat:@"%@_chunk%dof%d.mlmodelc", basename, chunk_idx + 1, num_chunks];
            NSString *model_path = [path_model_str stringByAppendingPathComponent:model_name];
            NSURL *url_model = [NSURL fileURLWithPath:model_path];

            error = nil;
            auto decode_start = std::chrono::steady_clock::now();
            MLModel *mlmodel_decode = [MLModel modelWithContentsOfURL:url_model configuration:config_decode error:&error];
            auto decode_end = std::chrono::steady_clock::now();
            double decode_ms = std::chrono::duration<double, std::milli>(decode_end - decode_start).count();
            NSLog(@"Loaded chunk %d/%d (%@) decode: %.2f ms",
                  chunk_idx + 1, num_chunks, model_name, decode_ms);
            if (error || !mlmodel_decode) {
                NSLog(@"Error loading decode model %@: %@", model_name, error);
                rwkv_coreml_release_resources(ctx);
                return -1;
            }
            ctx->load_done_chunks = chunk_idx * 2 + 1;

            error = nil;
            auto prefill_start = std::chrono::steady_clock::now();
            MLModel *mlmodel_prefill = [MLModel modelWithContentsOfURL:url_model configuration:config_prefill error:&error];
            auto prefill_end = std::chrono::steady_clock::now();
            double prefill_ms = std::chrono::duration<double, std::milli>(prefill_end - prefill_start).count();
            NSLog(@"Loaded chunk %d/%d (%@) prefill: %.2f ms",
                  chunk_idx + 1, num_chunks, model_name, prefill_ms);
            if (error || !mlmodel_prefill) {
                NSLog(@"Error loading prefill model %@: %@", model_name, error);
                rwkv_coreml_release_resources(ctx);
                return -1;
            }
            ctx->load_done_chunks = chunk_idx * 2 + 2;
            if (num_chunks == 1) {
                rwkv_coreml_singlechunk_impl *model_decode = [[rwkv_coreml_singlechunk_impl alloc] initWithMLModel:mlmodel_decode];
                rwkv_coreml_singlechunk_impl *model_prefill = [[rwkv_coreml_singlechunk_impl alloc] initWithMLModel:mlmodel_prefill];
                ctx->model_decode.push_back(CFBridgingRetain(model_decode));
                ctx->model_prefill.push_back(CFBridgingRetain(model_prefill));
                rwkv_coreml_singlechunk_implState *state = [model_decode newState];
                ctx->states.push_back(CFBridgingRetain(state));
            } else if (chunk_idx == 0) {
                rwkv_coreml_firstchunk_impl *model_decode = [[rwkv_coreml_firstchunk_impl alloc] initWithMLModel:mlmodel_decode];
                rwkv_coreml_firstchunk_impl *model_prefill = [[rwkv_coreml_firstchunk_impl alloc] initWithMLModel:mlmodel_prefill];
                ctx->model_decode.push_back(CFBridgingRetain(model_decode));
                ctx->model_prefill.push_back(CFBridgingRetain(model_prefill));
                rwkv_coreml_firstchunk_implState *state = [model_decode newState];
                ctx->states.push_back(CFBridgingRetain(state));
            } else {
                rwkv_coreml_impl *model_decode = [[rwkv_coreml_impl alloc] initWithMLModel:mlmodel_decode];
                rwkv_coreml_impl *model_prefill = [[rwkv_coreml_impl alloc] initWithMLModel:mlmodel_prefill];
                ctx->model_decode.push_back(CFBridgingRetain(model_decode));
                ctx->model_prefill.push_back(CFBridgingRetain(model_prefill));
                rwkv_coreml_implState *state = [model_decode newState];
                ctx->states.push_back(CFBridgingRetain(state));
            }

            if (chunk_idx == 0) {
                NSDictionary *model_inputs_prefill = mlmodel_prefill.modelDescription.inputDescriptionsByName;
                NSArray<NSNumber *> *in_prefill_shape = get_shape_by_name(model_inputs_prefill, @"in0");
                if (in_prefill_shape == nil) {
                    NSLog(@"Error getting in_prefill shape");
                    rwkv_coreml_release_resources(ctx);
                    return -1;
                }
                prefill_seq_length = [in_prefill_shape[1] intValue];
            }

            NSDictionary *model_outputs = mlmodel_decode.modelDescription.outputDescriptionsByName;
            if (chunk_idx == num_chunks - 1) {
                NSArray<NSNumber *> *logits_out_shape = get_shape_by_name(model_outputs, @"out0");
                if (logits_out_shape == nil) {
                    NSLog(@"Error getting out0 shape");
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
                NSLog(@"Error getting state_wkv shape");
                rwkv_coreml_release_resources(ctx);
                return -1;
            }
            total_layers += [state_wkv_shape[0] intValue];
            int chunk_num_heads = [state_wkv_shape[1] intValue];
            int chunk_head_dim = [state_wkv_shape[2] intValue];
            if (num_heads == 0) num_heads = chunk_num_heads;
            if (head_dim == 0) head_dim = chunk_head_dim;
            if (num_heads != chunk_num_heads || head_dim != chunk_head_dim) {
                NSLog(@"Warning: inconsistent head shape in chunk %d (heads=%d dim=%d)", chunk_idx, chunk_num_heads, chunk_head_dim);
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

        ctx->prefill_seq_length = prefill_seq_length;
        ctx->n_layers = total_layers;
        ctx->num_heads = num_heads;
        ctx->head_dim = head_dim;
        ctx->embd_dim = ctx->head_dim * ctx->num_heads;
        ctx->vocab_size = vocab_size;

        auto total_end = std::chrono::steady_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();
        NSLog(@"Total model load time: %.2f ms", total_ms);

        ctx->state_wkv_bytes = 0;
        ctx->state_tokenshift_bytes = 0;
        for (int i = 0; i < num_chunks; ++i) {
            ctx->state_wkv_bytes += ctx->state_wkv_bytes_per_chunk[i];
            ctx->state_tokenshift_bytes += ctx->state_tokenshift_bytes_per_chunk[i];
        }

        NSLog(@"num_chunks: %d, num_heads: %d, head_dim: %d, vocab_size: %d, n_layers: %d, prefill_seq_length: %d, state_wkv_bytes: %zu, state_tokenshift_bytes: %zu\n",
            ctx->num_chunks, ctx->num_heads, ctx->head_dim, ctx->vocab_size, ctx->n_layers, ctx->prefill_seq_length, ctx->state_wkv_bytes, ctx->state_tokenshift_bytes);
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
    int total_steps = std::max(1, ctx->num_chunks * 2);
    int done_steps = std::max(0, std::min(total_steps, ctx->load_done_chunks));
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
        if (tokens.size() != ctx->prefill_seq_length) {
            NSLog(@"Error: tokens size is not equal to prefill_seq_length");
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
    return ctx->prefill_seq_length;
}

int rwkv_coreml_get_state_wkv_bytes(struct rwkv_coreml_context * ctx) {
    return ctx->state_wkv_bytes;
}

int rwkv_coreml_get_state_tokenshift_bytes(struct rwkv_coreml_context * ctx) {
    return ctx->state_tokenshift_bytes;
}

std::vector<std::vector<uint8_t>> rwkv_coreml_get_state(struct rwkv_coreml_context * ctx) {
    std::vector<std::vector<uint8_t>> state_ret(2); // wkv and tokenshift
    if (!ctx || ctx->states.empty()) {
        NSLog(@"rwkv_coreml_get_state: invalid ctx/state");
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
                    NSLog(@"rwkv_coreml_get_state: state_wkv dst buffer is empty (init-time size not captured?) src=%zu", src_size);
                    return;
                }
                if (src_size != wkv_bytes) {
                    NSLog(@"rwkv_coreml_get_state: state_wkv size mismatch: src=%zu expected=%zu", src_size, wkv_bytes);
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
                    NSLog(@"rwkv_coreml_get_state: state_tokenshift dst buffer is empty (init-time size not captured?) src=%zu", src_size);
                    return;
                }
                if (src_size != tokenshift_bytes) {
                    NSLog(@"rwkv_coreml_get_state: state_tokenshift size mismatch: src=%zu expected=%zu", src_size, tokenshift_bytes);
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
    if (!ctx || ctx->states.empty()) {
        NSLog(@"rwkv_coreml_set_state: invalid ctx/state");
        return;
    }
    if (state.size() < 2) {
        NSLog(@"rwkv_coreml_set_state: invalid state vector size: %zu", state.size());
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
                    NSLog(@"rwkv_coreml_set_state: state_wkv size mismatch: dst=%zu expected=%zu", dst_size, wkv_bytes);
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
                    NSLog(@"rwkv_coreml_set_state: state_tokenshift size mismatch: dst=%zu expected=%zu", dst_size, tokenshift_bytes);
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
    if (!ctx || ctx->states.empty()) {
        NSLog(@"rwkv_coreml_set_wkv_state: invalid ctx/state");
        return;
    }
    if (state.size() * sizeof(half_float::half) != ctx->state_wkv_bytes) {
        NSLog(@"rwkv_coreml_set_wkv_state: invalid state vector size: %zu", state.size() * sizeof(half_float::half));
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
    if (!ctx || ctx->states.empty()) return;
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