//
// lmhead.h
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
@interface lmheadInput : NSObject<MLFeatureProvider>

/// in0 as 1 × 1 × 768 3-dimensional array of 16-bit floats
@property (readwrite, nonatomic, strong) MLMultiArray * in0;
- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithIn0:(MLMultiArray *)in0 NS_DESIGNATED_INITIALIZER;

@end

/// Model Prediction Output Type
API_AVAILABLE(macos(15.0), ios(18.0), watchos(11.0), tvos(18.0)) __attribute__((visibility("hidden")))
@interface lmheadOutput : NSObject<MLFeatureProvider>

/// out0 as 1 × 1 × 65536 3-dimensional array of 16-bit floats
@property (readwrite, nonatomic, strong) MLMultiArray * out0;
- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithOut0:(MLMultiArray *)out0 NS_DESIGNATED_INITIALIZER;

@end

/// Class for model loading and prediction
API_AVAILABLE(macos(15.0), ios(18.0), watchos(11.0), tvos(18.0)) __attribute__((visibility("hidden")))
@interface lmhead : NSObject
@property (readonly, nonatomic, nullable) MLModel * model;

/**
    URL of the underlying .mlmodelc directory.
*/
+ (nullable NSURL *)URLOfModelInThisBundle;

/**
    Initialize lmhead instance from an existing MLModel object.

    Usually the application does not use this initializer unless it makes a subclass of lmhead.
    Such application may want to use `-[MLModel initWithContentsOfURL:configuration:error:]` and `+URLOfModelInThisBundle` to create a MLModel object to pass-in.
*/
- (instancetype)initWithMLModel:(MLModel *)model NS_DESIGNATED_INITIALIZER;

/**
    Initialize lmhead instance with the model in this bundle.
*/
- (nullable instancetype)init;

/**
    Initialize lmhead instance with the model in this bundle.

    @param configuration The model configuration object
    @param error If an error occurs, upon return contains an NSError object that describes the problem. If you are not interested in possible errors, pass in NULL.
*/
- (nullable instancetype)initWithConfiguration:(MLModelConfiguration *)configuration error:(NSError * _Nullable __autoreleasing * _Nullable)error;

/**
    Initialize lmhead instance from the model URL.

    @param modelURL URL to the .mlmodelc directory for lmhead.
    @param error If an error occurs, upon return contains an NSError object that describes the problem. If you are not interested in possible errors, pass in NULL.
*/
- (nullable instancetype)initWithContentsOfURL:(NSURL *)modelURL error:(NSError * _Nullable __autoreleasing * _Nullable)error;

/**
    Initialize lmhead instance from the model URL.

    @param modelURL URL to the .mlmodelc directory for lmhead.
    @param configuration The model configuration object
    @param error If an error occurs, upon return contains an NSError object that describes the problem. If you are not interested in possible errors, pass in NULL.
*/
- (nullable instancetype)initWithContentsOfURL:(NSURL *)modelURL configuration:(MLModelConfiguration *)configuration error:(NSError * _Nullable __autoreleasing * _Nullable)error;

/**
    Construct lmhead instance asynchronously with configuration.
    Model loading may take time when the model content is not immediately available (e.g. encrypted model). Use this factory method especially when the caller is on the main thread.

    @param configuration The model configuration
    @param handler When the model load completes successfully or unsuccessfully, the completion handler is invoked with a valid lmhead instance or NSError object.
*/
+ (void)loadWithConfiguration:(MLModelConfiguration *)configuration completionHandler:(void (^)(lmhead * _Nullable model, NSError * _Nullable error))handler;

/**
    Construct lmhead instance asynchronously with URL of .mlmodelc directory and optional configuration.

    Model loading may take time when the model content is not immediately available (e.g. encrypted model). Use this factory method especially when the caller is on the main thread.

    @param modelURL The model URL.
    @param configuration The model configuration
    @param handler When the model load completes successfully or unsuccessfully, the completion handler is invoked with a valid lmhead instance or NSError object.
*/
+ (void)loadContentsOfURL:(NSURL *)modelURL configuration:(MLModelConfiguration *)configuration completionHandler:(void (^)(lmhead * _Nullable model, NSError * _Nullable error))handler;

/**
    Make a prediction using the standard interface
    @param input an instance of lmheadInput to predict from
    @param error If an error occurs, upon return contains an NSError object that describes the problem. If you are not interested in possible errors, pass in NULL.
    @return the prediction as lmheadOutput
*/
- (nullable lmheadOutput *)predictionFromFeatures:(lmheadInput *)input error:(NSError * _Nullable __autoreleasing * _Nullable)error;

/**
    Make a prediction using the standard interface
    @param input an instance of lmheadInput to predict from
    @param options prediction options
    @param error If an error occurs, upon return contains an NSError object that describes the problem. If you are not interested in possible errors, pass in NULL.
    @return the prediction as lmheadOutput
*/
- (nullable lmheadOutput *)predictionFromFeatures:(lmheadInput *)input options:(MLPredictionOptions *)options error:(NSError * _Nullable __autoreleasing * _Nullable)error;

/**
    Make an asynchronous prediction using the standard interface
    @param input an instance of lmheadInput to predict from
    @param completionHandler a block that will be called upon completion of the prediction. error will be nil if no error occurred.
*/
- (void)predictionFromFeatures:(lmheadInput *)input completionHandler:(void (^)(lmheadOutput * _Nullable output, NSError * _Nullable error))completionHandler;

/**
    Make an asynchronous prediction using the standard interface
    @param input an instance of lmheadInput to predict from
    @param options prediction options
    @param completionHandler a block that will be called upon completion of the prediction. error will be nil if no error occurred.
*/
- (void)predictionFromFeatures:(lmheadInput *)input options:(MLPredictionOptions *)options completionHandler:(void (^)(lmheadOutput * _Nullable output, NSError * _Nullable error))completionHandler;

/**
    Make a prediction using the convenience interface
    @param in0 1 × 1 × 768 3-dimensional array of 16-bit floats
    @param error If an error occurs, upon return contains an NSError object that describes the problem. If you are not interested in possible errors, pass in NULL.
    @return the prediction as lmheadOutput
*/
- (nullable lmheadOutput *)predictionFromIn0:(MLMultiArray *)in0 error:(NSError * _Nullable __autoreleasing * _Nullable)error;

/**
    Batch prediction
    @param inputArray array of lmheadInput instances to obtain predictions from
    @param options prediction options
    @param error If an error occurs, upon return contains an NSError object that describes the problem. If you are not interested in possible errors, pass in NULL.
    @return the predictions as NSArray<lmheadOutput *>
*/
- (nullable NSArray<lmheadOutput *> *)predictionsFromInputs:(NSArray<lmheadInput*> *)inputArray options:(MLPredictionOptions *)options error:(NSError * _Nullable __autoreleasing * _Nullable)error;
@end

NS_ASSUME_NONNULL_END
