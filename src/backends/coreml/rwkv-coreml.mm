#if !__has_feature(objc_arc)
#error This file must be compiled with automatic reference counting enabled (-fobjc-arc)
#endif

#import "rwkv-coreml.h"
#import "rwkv-coreml-stateful-impl.h"

#import <CoreML/CoreML.h>

#include <stdlib.h>
#include <cstdio>
#include <vector>
#include "half.hpp"

struct rwkv_coreml_context {
    const void * model_decode;
    const void * model_prefill;
    int n_layers;
    int num_heads;
    int head_dim;
    int embd_dim;
    int vocab_size;
    int prefill_seq_length;

    rwkv_coreml_stateful_implState * state = nullptr;
    rwkv_coreml_stateful_implOutput * out_prefill = nullptr;
    rwkv_coreml_stateful_implOutput * out_decode = nullptr;
};

NSArray<NSNumber *> * get_shape_by_name(NSDictionary *model_inputs, NSString *name) {
    MLFeatureDescription *desc = model_inputs[name];
    if (desc.type == MLFeatureTypeMultiArray) {
        return desc.multiArrayConstraint.shape;
    }
    return nil;
}

struct rwkv_coreml_context * rwkv_coreml_init(const char * path_model) {
    NSString * path_model_str = [[NSString alloc] initWithUTF8String:path_model];

    NSURL * url_model = [NSURL fileURLWithPath: path_model_str];

    // select which device to run the Core ML model on
    MLModelConfiguration *config_decode = [[MLModelConfiguration alloc] init];
    config_decode.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
    config_decode.functionName = @"decode";

    MLModelConfiguration *config_prefill = [[MLModelConfiguration alloc] init];
    config_prefill.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
    config_prefill.functionName = @"prefill";

    NSError *error = nil;

    MLModel *mlmodel_decode = [MLModel modelWithContentsOfURL:url_model configuration:config_decode error:&error];
    MLModel *mlmodel_prefill = [MLModel modelWithContentsOfURL:url_model configuration:config_prefill error:&error];

    if (error || !mlmodel_decode || !mlmodel_prefill) {
        NSLog(@"Error loading model: %@", error);
        return NULL;
    }

    rwkv_coreml_context * ctx = new rwkv_coreml_context;

    NSDictionary *model_inputs = mlmodel_decode.modelDescription.inputDescriptionsByName;
    NSDictionary *model_inputs_prefill = mlmodel_prefill.modelDescription.inputDescriptionsByName;
    NSDictionary *model_outputs = mlmodel_decode.modelDescription.outputDescriptionsByName;
    int num_inputs = mlmodel_decode.modelDescription.inputDescriptionsByName.count;

    if (num_inputs != 1) {
        NSLog(@"only support stateful model");
        return NULL;
    }

    rwkv_coreml_stateful_impl * model_decode = [[rwkv_coreml_stateful_impl alloc] initWithMLModel:mlmodel_decode];
    ctx->model_decode = CFBridgingRetain(model_decode);

    rwkv_coreml_stateful_impl * model_prefill = [[rwkv_coreml_stateful_impl alloc] initWithMLModel:mlmodel_prefill];
    ctx->model_prefill = CFBridgingRetain(model_prefill);

    ctx->state = [model_decode newState];
    __block MLMultiArray *state_wkv;
    [ctx->state getMultiArrayForState:rwkv_coreml_stateful_implStateNameState_wkv handler:^(MLMultiArray *buffer) {
        state_wkv = buffer;
    }];
    NSArray<NSNumber *> *state_wkv_shape = state_wkv.shape;
    if (state_wkv_shape == nil) {
        NSLog(@"Error getting state_wkv shape");
        return NULL;
    }
    NSArray<NSNumber *> *in_prefill_shape = get_shape_by_name(model_inputs_prefill, @"in0");
    if (in_prefill_shape == nil) {
        NSLog(@"Error getting in_prefill shape");
        return NULL;
    }
    ctx->prefill_seq_length = [in_prefill_shape[1] intValue];
    ctx->n_layers = [state_wkv_shape[0] intValue];
    ctx->num_heads = [state_wkv_shape[1] intValue];
    ctx->head_dim = [state_wkv_shape[2] intValue];
    ctx->embd_dim = ctx->head_dim * ctx->num_heads;

    if (ctx->model_decode == NULL || ctx->model_prefill == NULL) {
        NSLog(@"Error loading model");
        return NULL;
    }

    NSArray<NSNumber *> *logits_out_shape = get_shape_by_name(model_outputs, @"logits");
    if (logits_out_shape == nil) {
        NSLog(@"Error getting logits shape");
        return NULL;
    }
    ctx->vocab_size = [logits_out_shape[2] intValue];

    NSLog(@"num_heads: %d, head_dim: %d, vocab_size: %d, n_layers: %d, prefill_seq_length: %d\n", ctx->num_heads, ctx->head_dim, ctx->vocab_size, ctx->n_layers, ctx->prefill_seq_length);

    return ctx;
}

void rwkv_coreml_free(struct rwkv_coreml_context * ctx) {
    if (ctx) {
        if (ctx->model_decode) {
            CFRelease(ctx->model_decode);
        }
        if (ctx->model_prefill) {
            CFRelease(ctx->model_prefill);
        }
        delete ctx;
    }
}

void* rwkv_coreml_decode(struct rwkv_coreml_context * ctx, int token) {
    float token_float = (float)token;
    MLMultiArray * inMultiArray = [
        [MLMultiArray alloc] initWithDataPointer: &token
                                           shape: @[@1, @(1)]
                                        dataType: MLMultiArrayDataTypeInt32
                                         strides: @[@(1), @(1)]
                                     deallocator: nil
                                           error: nil
    ];

    if (!ctx->state) {
        ctx->state = [(__bridge rwkv_coreml_stateful_impl *) ctx->model_decode newState];
    }

    ctx->out_decode = [(__bridge id) ctx->model_decode predictionFromIn0: inMultiArray usingState: ctx->state error: nil];
    return ctx->out_decode.logits.dataPointer;
}

void* rwkv_coreml_prefill(struct rwkv_coreml_context * ctx, std::vector<int> tokens) {
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

    if (!ctx->state) {
        ctx->state = [(__bridge rwkv_coreml_stateful_impl *) ctx->model_decode newState];
    }

    ctx->out_prefill = [(__bridge id) ctx->model_prefill predictionFromIn0: inMultiArray usingState: ctx->state error: nil];
    return ctx->out_prefill.logits.dataPointer;
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

std::vector<std::vector<uint8_t>> rwkv_coreml_get_state(struct rwkv_coreml_context * ctx) {
    std::vector<std::vector<uint8_t>> state;

    // TODO: read stateful model's state
    return state;
}

void rwkv_coreml_set_state(struct rwkv_coreml_context * ctx, std::vector<std::vector<uint8_t>> state) {
    // TODO: set stateful model's state
}

void rwkv_coreml_zero_state(struct rwkv_coreml_context * ctx) {
    ctx->state = [(__bridge rwkv_coreml_stateful_impl *) ctx->model_decode newState];
}