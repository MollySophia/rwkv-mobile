//
// rwkv_coreml_impl.h
//
// This file was automatically generated and should not be edited.
//

#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#include <stdint.h>
#include <os/log.h>

NS_ASSUME_NONNULL_BEGIN

/// Model Prediction Input Type
API_AVAILABLE(macos(15.0), ios(18.0), watchos(11.0), tvos(18.0)) __attribute__((visibility("hidden")))
@interface rwkv_coreml_implInput : NSObject<MLFeatureProvider>

/// in0 as 1 × 1 × 768 3-dimensional array of 16-bit floats
@property (readwrite, nonatomic, strong) MLMultiArray * in0;

/// v_first_in as 1 × 1 × 768 3-dimensional array of 16-bit floats
@property (readwrite, nonatomic, strong) MLMultiArray * v_first_in;
- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithIn0:(MLMultiArray *)in0 v_first_in:(MLMultiArray *)v_first_in NS_DESIGNATED_INITIALIZER;

@end

/// Model Prediction Output Type
API_AVAILABLE(macos(15.0), ios(18.0), watchos(11.0), tvos(18.0)) __attribute__((visibility("hidden")))
@interface rwkv_coreml_implOutput : NSObject<MLFeatureProvider>

/// out0 as 1 × 1 × 768 3-dimensional array of 16-bit floats
@property (readwrite, nonatomic, strong) MLMultiArray * out0;
- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithOut0:(MLMultiArray *)out0 NS_DESIGNATED_INITIALIZER;

@end

/// Model Prediction State Names
typedef NS_ENUM(NSInteger, rwkv_coreml_implStateName) {
    rwkv_coreml_implStateNameState_tokenshift,
    rwkv_coreml_implStateNameState_wkv,
} API_AVAILABLE(macos(15.0), ios(18.0), watchos(11.0), tvos(18.0)) __attribute__((visibility("hidden")));

/// Model Prediction State Type
API_AVAILABLE(macos(15.0), ios(18.0), watchos(11.0), tvos(18.0)) __attribute__((visibility("hidden")))
@interface rwkv_coreml_implState : NSObject
@property (readonly, strong, nonatomic) MLState *state;
- (void)getMultiArrayForState:(rwkv_coreml_implStateName)stateName
                      handler:(void (NS_NOESCAPE ^)(MLMultiArray *buffer))handler;
@end

/// Class for model loading and prediction
API_AVAILABLE(macos(15.0), ios(18.0), watchos(11.0), tvos(18.0)) __attribute__((visibility("hidden")))
@interface rwkv_coreml_impl : NSObject
@property (readonly, nonatomic, nullable) MLModel * model;

/**
    URL of the underlying .mlmodelc directory.
*/
+ (nullable NSURL *)URLOfModelInThisBundle;

/**
    Initialize rwkv_coreml_impl instance from an existing MLModel object.

    Usually the application does not use this initializer unless it makes a subclass of rwkv_coreml_impl.
    Such application may want to use `-[MLModel initWithContentsOfURL:configuration:error:]` and `+URLOfModelInThisBundle` to create a MLModel object to pass-in.
*/
- (instancetype)initWithMLModel:(MLModel *)model NS_DESIGNATED_INITIALIZER;

/**
    Initialize rwkv_coreml_impl instance with the model in this bundle.
*/
- (nullable instancetype)init;

/**
    Initialize rwkv_coreml_impl instance with the model in this bundle.

    @param configuration The model configuration object
    @param error If an error occurs, upon return contains an NSError object that describes the problem. If you are not interested in possible errors, pass in NULL.
*/
- (nullable instancetype)initWithConfiguration:(MLModelConfiguration *)configuration error:(NSError * _Nullable __autoreleasing * _Nullable)error;

/**
    Initialize rwkv_coreml_impl instance from the model URL.

    @param modelURL URL to the .mlmodelc directory for rwkv_coreml_impl.
    @param error If an error occurs, upon return contains an NSError object that describes the problem. If you are not interested in possible errors, pass in NULL.
*/
- (nullable instancetype)initWithContentsOfURL:(NSURL *)modelURL error:(NSError * _Nullable __autoreleasing * _Nullable)error;

/**
    Initialize rwkv_coreml_impl instance from the model URL.

    @param modelURL URL to the .mlmodelc directory for rwkv_coreml_impl.
    @param configuration The model configuration object
    @param error If an error occurs, upon return contains an NSError object that describes the problem. If you are not interested in possible errors, pass in NULL.
*/
- (nullable instancetype)initWithContentsOfURL:(NSURL *)modelURL configuration:(MLModelConfiguration *)configuration error:(NSError * _Nullable __autoreleasing * _Nullable)error;

/**
    Construct rwkv_coreml_impl instance asynchronously with configuration.
    Model loading may take time when the model content is not immediately available (e.g. encrypted model). Use this factory method especially when the caller is on the main thread.

    @param configuration The model configuration
    @param handler When the model load completes successfully or unsuccessfully, the completion handler is invoked with a valid rwkv_coreml_impl instance or NSError object.
*/
+ (void)loadWithConfiguration:(MLModelConfiguration *)configuration completionHandler:(void (^)(rwkv_coreml_impl * _Nullable model, NSError * _Nullable error))handler;

/**
    Construct rwkv_coreml_impl instance asynchronously with URL of .mlmodelc directory and optional configuration.

    Model loading may take time when the model content is not immediately available (e.g. encrypted model). Use this factory method especially when the caller is on the main thread.

    @param modelURL The model URL.
    @param configuration The model configuration
    @param handler When the model load completes successfully or unsuccessfully, the completion handler is invoked with a valid rwkv_coreml_impl instance or NSError object.
*/
+ (void)loadContentsOfURL:(NSURL *)modelURL configuration:(MLModelConfiguration *)configuration completionHandler:(void (^)(rwkv_coreml_impl * _Nullable model, NSError * _Nullable error))handler;

/**
    Make a new state.
*/
- (rwkv_coreml_implState *)newState;

/**
    Make a prediction using the standard interface
    @param input an instance of rwkv_coreml_implInput to predict from
    @param state prediction state
    @param error If an error occurs, upon return contains an NSError object that describes the problem. If you are not interested in possible errors, pass in NULL.
    @return the prediction as rwkv_coreml_implOutput
*/
- (nullable rwkv_coreml_implOutput *)predictionFromFeatures:(rwkv_coreml_implInput *)input usingState:(rwkv_coreml_implState *)state error:(NSError * _Nullable __autoreleasing * _Nullable)error;

/**
    Make a prediction using the standard interface
    @param input an instance of rwkv_coreml_implInput to predict from
    @param state prediction state
    @param options prediction options
    @param error If an error occurs, upon return contains an NSError object that describes the problem. If you are not interested in possible errors, pass in NULL.
    @return the prediction as rwkv_coreml_implOutput
*/
- (nullable rwkv_coreml_implOutput *)predictionFromFeatures:(rwkv_coreml_implInput *)input usingState:(rwkv_coreml_implState *)state options:(MLPredictionOptions *)options error:(NSError * _Nullable __autoreleasing * _Nullable)error;

/**
    Make an asynchronous prediction using the standard interface
    @param input an instance of rwkv_coreml_implInput to predict from
    @param state prediction state
    @param completionHandler a block that will be called upon completion of the prediction. error will be nil if no error occurred.
*/
- (void)predictionFromFeatures:(rwkv_coreml_implInput *)input usingState:(rwkv_coreml_implState *)state completionHandler:(void (^)(rwkv_coreml_implOutput * _Nullable output, NSError * _Nullable error))completionHandler;

/**
    Make an asynchronous prediction using the standard interface
    @param input an instance of rwkv_coreml_implInput to predict from
    @param state prediction state
    @param options prediction options
    @param completionHandler a block that will be called upon completion of the prediction. error will be nil if no error occurred.
*/
- (void)predictionFromFeatures:(rwkv_coreml_implInput *)input usingState:(rwkv_coreml_implState *)state options:(MLPredictionOptions *)options completionHandler:(void (^)(rwkv_coreml_implOutput * _Nullable output, NSError * _Nullable error))completionHandler;

/**
    Make a prediction using the convenience interface
    @param in0 1 × 1 × 768 3-dimensional array of 16-bit floats
    @param v_first_in 1 × 1 × 768 3-dimensional array of 16-bit floats
    @param state prediction state
    @param error If an error occurs, upon return contains an NSError object that describes the problem. If you are not interested in possible errors, pass in NULL.
    @return the prediction as rwkv_coreml_implOutput
*/
- (nullable rwkv_coreml_implOutput *)predictionFromIn0:(MLMultiArray *)in0 v_first_in:(MLMultiArray *)v_first_in usingState:(rwkv_coreml_implState *)state error:(NSError * _Nullable __autoreleasing * _Nullable)error;
@end

NS_ASSUME_NONNULL_END
