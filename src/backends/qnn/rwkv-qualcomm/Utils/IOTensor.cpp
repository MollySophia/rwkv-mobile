//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================
#include <cstring>
#include <fstream>
#include <iostream>

#include "ClientBuffer.hpp"
#ifndef _WIN32
#include "DmaBufAllocator.hpp"
#endif
#include "IBufferAlloc.hpp"
#include "IOTensor.hpp"
#include "QnnTypeMacros.hpp"
#include "RpcMem.hpp"
#include "logger.h"
#ifdef _WIN32
#define __strdup _strdup
#else
#define __strdup strdup
#endif

IOTensor::IOTensor(BufferAlloc bufferAllocIn, QNN_INTERFACE_VER_TYPE* qnnInterface)
    : m_bufferAlloc(bufferAllocIn),
      m_qnnInterface(qnnInterface),
      m_bufferManager(new ClientBuffer()) {}

bool IOTensor::initialize(Qnn_ContextHandle_t contextHandle) {
  if (m_bufferAlloc == BufferAlloc::SHARED_BUFFER) {
    rwkvmobile::LOGI("[QNN] Using RpcMem as buffer manager");
    m_bufferManager = std::unique_ptr<IBufferAlloc>(new RpcMem(contextHandle, m_qnnInterface));
  } else if (m_bufferAlloc == BufferAlloc::DMABUF) {
    rwkvmobile::LOGI("[QNN] Using QnnDmaBufferAllocator as buffer manager");
#ifdef _WIN32
    return false;
#else
    m_bufferManager =
        std::unique_ptr<IBufferAlloc>(new QnnDmaBufferAllocator(contextHandle, m_qnnInterface));
#endif
  }

  if (true != m_bufferManager->initialize()) {
    rwkvmobile::LOGE("Failed to initialize buffer manager");
    return false;
  }

  return true;
}

IOTensor::~IOTensor() {
  if (m_bufferAlloc == BufferAlloc::SHARED_BUFFER || m_bufferAlloc == BufferAlloc::DMABUF) {
    m_bufferManager->freeFusedBuffers();
  }
}

// Setup details for Qnn_Tensor_t for execution
// based on information in TensorWrapper provided by model.so.
bool IOTensor::setupTensors(Qnn_Tensor_t** tensors,
                            std::unordered_map<std::string, void*>& tensorNameToTensorPointer,
                            uint32_t tensorCount,
                            TensorWrapper* tensorWrappers,
                            std::unordered_map<std::string, size_t>& tensorsSize,
                            Qnn_ContextHandle_t contextHandle,
                            bool skipBufferAllocation) {
  if (nullptr == tensorWrappers) {
    rwkvmobile::LOGE("tensorWrappers is nullptr");
    return false;
  }
  if (0 == tensorCount) {
    rwkvmobile::LOGD("tensor count is 0. Nothing to setup.");
    return true;
  }

  *tensors = (Qnn_Tensor_t*)calloc(1, tensorCount * sizeof(Qnn_Tensor_t));
  if (nullptr == *tensors) {
    rwkvmobile::LOGE("mem alloc failed for *tensors");
    return false;
  }

  auto returnStatus = true;

  uint64_t totalBufferSize = 0;
  void* memPointer         = nullptr;
  int32_t fd               = -1;
  if (m_bufferAlloc == BufferAlloc::SHARED_BUFFER) {
    // Calculate the total size of the tensors
    for (size_t tensorIdx = 0; tensorIdx < tensorCount; tensorIdx++) {
      auto wrapperTensorName = std::string(GET_TENSOR_WRAPPER_NAME(tensorWrappers[tensorIdx]));
      totalBufferSize += tensorsSize[wrapperTensorName];
    }
    rwkvmobile::LOGD("Calculated total size %lu", totalBufferSize);

    if (!skipBufferAllocation) {
      // Allocate the buffer of this size
      memPointer = m_bufferManager->allocateTensorFusedBuffer(totalBufferSize, &fd);
      if (memPointer) {
        rwkvmobile::LOGD("Successfully allocated a buffer of size %lu, pointer %p, fd %d",
                  (unsigned long)totalBufferSize,
                  memPointer,
                  fd);
      } else {
        rwkvmobile::LOGE("Not able to allocate buffer of size %lu", (unsigned long)totalBufferSize);
        return false;
      }
    }
  }

  uint64_t offset = 0;

  for (size_t tensorIdx = 0; tensorIdx < tensorCount; tensorIdx++) {
    Qnn_Tensor_t wrapperTensor = GET_TENSOR_WRAPPER_TENSOR(tensorWrappers[tensorIdx]);
    auto wrapperTensorName     = std::string(GET_TENSOR_WRAPPER_NAME(tensorWrappers[tensorIdx]));
    if (true == returnStatus) {
      (*tensors)[tensorIdx] = QNN_TENSOR_INIT;
      returnStatus          = deepCopyQnnTensorInfo(((*tensors) + tensorIdx), &wrapperTensor);
    }
    if (true == returnStatus) {
      size_t tensorDataSize = tensorsSize[wrapperTensorName];
      if (m_bufferAlloc == BufferAlloc::SHARED_BUFFER) {
        if (!skipBufferAllocation) {
          returnStatus = m_bufferManager->mapFusedBufferOffset(((*tensors) + tensorIdx),
                                                               tensorDataSize,
                                                               fd,
                                                               offset,
                                                               totalBufferSize,
                                                               memPointer,
                                                               contextHandle);
          offset += tensorDataSize;
        }
      } else {
        returnStatus =
            m_bufferManager->allocateTensorBuffer(((*tensors) + tensorIdx), tensorDataSize);
      }
    }
    if (true != returnStatus) {
      rwkvmobile::LOGE("Failure in setupTensors, cleaning up resources");
      tearDownTensors(*tensors, tensorIdx);
      *tensors = nullptr;
      rwkvmobile::LOGE("Failure in setupTensors, done cleaning up resources");
      return false;
    } else {
      tensorNameToTensorPointer.insert({wrapperTensorName, ((*tensors) + tensorIdx)});
      // LOGD("allocateBuffer successful");
    }
  }

  return returnStatus;
}

// Setup details for all input tensors for graph execution.
bool IOTensor::setupInputTensors(Qnn_Tensor_t** inputs,
                                 std::unordered_map<std::string, void*>& tensorNameToTensorPointer,
                                 const GraphInfo_t& graphInfo,
                                 std::unordered_map<std::string, size_t>& inputTensorsSize,
                                 Qnn_ContextHandle_t contextHandle,
                                 bool skipBufferAllocation) {
  if (true != setupTensors(inputs,
                           tensorNameToTensorPointer,
                           graphInfo.numInputTensors,
                           (graphInfo.inputTensors),
                           inputTensorsSize,
                           contextHandle,
                           skipBufferAllocation)) {
    rwkvmobile::LOGE("Failure in setupInputTensors, cleaning up resources");
    if (nullptr != *inputs) {
      rwkvmobile::LOGD("cleaning up input tensors");
      tearDownTensors(*inputs, graphInfo.numInputTensors);
      *inputs = nullptr;
    }
    rwkvmobile::LOGE("Failure in setupInputTensors, done cleaning up resources");

    return false;
  }

  return true;
}

// Setup details for all output tensors for graph execution.
bool IOTensor::setupOutputTensors(Qnn_Tensor_t** outputs,
                                  std::unordered_map<std::string, void*>& tensorNameToTensorPointer,
                                  const GraphInfo_t& graphInfo,
                                  std::unordered_map<std::string, size_t>& outputTensorsSize,
                                  Qnn_ContextHandle_t contextHandle,
                                  bool skipBufferAllocation) {
  if (true != setupTensors(outputs,
                           tensorNameToTensorPointer,
                           graphInfo.numOutputTensors,
                           (graphInfo.outputTensors),
                           outputTensorsSize,
                           contextHandle,
                           skipBufferAllocation)) {
    rwkvmobile::LOGE("Failure in setupOutputTensors, cleaning up resources");
    if (nullptr != *outputs) {
      rwkvmobile::LOGD("cleaning up output tensors");
      tearDownTensors(*outputs, graphInfo.numOutputTensors);
      *outputs = nullptr;
    }
    rwkvmobile::LOGE("Failure in setupOutputTensors, done cleaning up resources");

    return false;
  }

  return true;
}

// Setup details for Qnn_Tensor_t for execution.
// Reuse same memory handle for KV input and input tensor.
bool IOTensor::setupInputWithSharedTensors(
    Qnn_Tensor_t** tensors,
    std::unordered_map<std::string, void*>& tensorNameToTensorPointer,
    const GraphInfo_t& graphInfo,
    std::unordered_map<std::string, size_t>& tensorsSize,
    Qnn_ContextHandle_t contextHandle,
    std::unordered_map<std::string, Qnn_Tensor_t*> sharedTensorMap) {
  uint32_t tensorCount          = graphInfo.numInputTensors;
  TensorWrapper* tensorWrappers = graphInfo.inputTensors;
  if (nullptr == tensorWrappers) {
    rwkvmobile::LOGE("tensorWrappers is nullptr");
    return false;
  }

  if (0 == tensorCount) {
    rwkvmobile::LOGD("tensor count is 0. Nothing to setup.");
    return true;
  }

  *tensors = (Qnn_Tensor_t*)calloc(1, tensorCount * sizeof(Qnn_Tensor_t));
  if (nullptr == *tensors) {
    rwkvmobile::LOGE("mem alloc failed for *tensors");
    return false;
  }

  bool returnStatus = true;
  for (size_t tensorIdx = 0; tensorIdx < tensorCount; tensorIdx++) {
    Qnn_Tensor_t wrapperTensor = GET_TENSOR_WRAPPER_TENSOR(tensorWrappers[tensorIdx]);
    auto wrapperTensorName     = std::string(GET_TENSOR_WRAPPER_NAME(tensorWrappers[tensorIdx]));
    if (true == returnStatus) {
      (*tensors)[tensorIdx] = QNN_TENSOR_INIT;
      returnStatus          = deepCopyQnnTensorInfo(((*tensors) + tensorIdx), &wrapperTensor);
    }
    if (true == returnStatus) {
      if (sharedTensorMap.find(wrapperTensorName) == sharedTensorMap.end()) {
        size_t tensorDataSize = tensorsSize[wrapperTensorName];
        rwkvmobile::LOGD("IoTensor :: Create Buffer for Tensor %s Size: %zu", wrapperTensorName.c_str(), tensorDataSize);
        returnStatus =
            m_bufferManager->allocateTensorBuffer(((*tensors) + tensorIdx), tensorDataSize);
      } else {
        std::string inputName = QNN_TENSOR_GET_NAME(sharedTensorMap[wrapperTensorName]);
        rwkvmobile::LOGD("IoTensor :: Reuse Buffer %s for Tensor %s",
                  inputName.c_str(),
                  wrapperTensorName.c_str());
        returnStatus = m_bufferManager->useSameMemory(((*tensors) + tensorIdx),
                                                      sharedTensorMap[wrapperTensorName]);
      }
    }
    if (true != returnStatus) {
      rwkvmobile::LOGE("Failure in setupTensors, cleaning up resources");
      tearDownTensors(*tensors, tensorIdx);
      *tensors = nullptr;
      rwkvmobile::LOGE("Failure in setupTensors, done cleaning up resources");
      break;
    } else {
      tensorNameToTensorPointer.insert({wrapperTensorName, ((*tensors) + tensorIdx)});
    }
  }
  return returnStatus;
}

bool IOTensor::setupOutputWithSharedTensors(
    Qnn_Tensor_t** tensors,
    std::unordered_map<std::string, void*>& tensorNameToTensorPointer,
    const GraphInfo_t& graphInfo,
    std::unordered_map<std::string, size_t>& tensorsSize,
    Qnn_ContextHandle_t contextHandle,
    std::unordered_map<std::string, Qnn_Tensor_t*> sharedTensorMap) {
  uint32_t tensorCount          = graphInfo.numOutputTensors;
  TensorWrapper* tensorWrappers = graphInfo.outputTensors;
  if (nullptr == tensorWrappers) {
    rwkvmobile::LOGE("tensorWrappers is nullptr");
    return false;
  }

  if (0 == tensorCount) {
    rwkvmobile::LOGD("tensor count is 0. Nothing to setup.");
    return true;
  }

  *tensors = (Qnn_Tensor_t*)calloc(1, tensorCount * sizeof(Qnn_Tensor_t));
  if (nullptr == *tensors) {
    rwkvmobile::LOGE("mem alloc failed for *tensors");
    return false;
  }

  bool returnStatus = true;
  for (size_t tensorIdx = 0; tensorIdx < tensorCount; tensorIdx++) {
    Qnn_Tensor_t wrapperTensor = GET_TENSOR_WRAPPER_TENSOR(tensorWrappers[tensorIdx]);
    auto wrapperTensorName     = std::string(GET_TENSOR_WRAPPER_NAME(tensorWrappers[tensorIdx]));
    if (true == returnStatus) {
      (*tensors)[tensorIdx] = QNN_TENSOR_INIT;
      returnStatus          = deepCopyQnnTensorInfo(((*tensors) + tensorIdx), &wrapperTensor);
    }
    if (true == returnStatus) {
      if (sharedTensorMap.find(wrapperTensorName) == sharedTensorMap.end()) {
        size_t tensorDataSize = tensorsSize[wrapperTensorName];
        rwkvmobile::LOGD("IoTensor :: Create Buffer for Tensor %s Size: %zu", wrapperTensorName.c_str(), tensorDataSize);
        returnStatus =
            m_bufferManager->allocateTensorBuffer(((*tensors) + tensorIdx), tensorDataSize);
      } else {
        std::string outputName = QNN_TENSOR_GET_NAME(sharedTensorMap[wrapperTensorName]);
        rwkvmobile::LOGD("IoTensor :: Reuse Buffer %s for Tensor %s",
                  outputName.c_str(),
                  wrapperTensorName.c_str());
        returnStatus = m_bufferManager->useSameMemory(((*tensors) + tensorIdx),
                                                      sharedTensorMap[wrapperTensorName]);
      }
    }
    if (true != returnStatus) {
      rwkvmobile::LOGE("Failure in setupTensors, cleaning up resources");
      tearDownTensors(*tensors, tensorIdx);
      *tensors = nullptr;
      rwkvmobile::LOGE("Failure in setupTensors, done cleaning up resources");
      break;
    } else {
      tensorNameToTensorPointer.insert({wrapperTensorName, ((*tensors) + tensorIdx)});
    }
  }
  return returnStatus;
}

bool IOTensor::mapFusedBufferOffset(
    GraphInfo_t* graph_info,
    Qnn_ContextHandle_t context_handle,
    const std::map<std::string, std::tuple<int, size_t, size_t>>& graph_allocs) {
  std::lock_guard lk(_tmp_lock);  // READ COMMENT IN IOTensor.hpp _tmp_lock

  bool ret = true;
  for (const bool mode : {true, false}) {
    TensorWrapper* tensor_bank = (mode) ? graph_info->inputTensors : graph_info->outputTensors;
    uint32_t num_tensors = (mode) ? graph_info->numInputTensors : graph_info->numOutputTensors;

    for (size_t tidx = 0; tidx < num_tensors; tidx++) {
      TensorWrapper& tensor_wrapper = tensor_bank[tidx];

      Qnn_Tensor_t* tensor    = &GET_TENSOR_WRAPPER_TENSOR(tensor_wrapper);
      std::string tensor_name = std::string(GET_TENSOR_WRAPPER_NAME(tensor_wrapper));

      if (graph_allocs.find(tensor_name) == graph_allocs.end()) continue;
      auto& [alloc_idx, offset, size] = graph_allocs.at(tensor_name);
      ret &= m_bufferManager->mapFusedBufferOffset(tensor, alloc_idx, offset, context_handle, size);
    }
  }

  return ret;
}

// Clean up all tensors related data after execution.
bool IOTensor::tearDownTensors(Qnn_Tensor_t* tensors, uint32_t tensorCount) {
  if (nullptr != tensors) {
    rwkvmobile::LOGD("cleaning up resources for tensors");
    for (size_t tensorIdx = 0; tensorIdx < tensorCount; tensorIdx++) {
      // LOGD("freeing resources for tensor: %zu", tensorIdx);
      if (nullptr != QNN_TENSOR_GET_DIMENSIONS(&tensors[tensorIdx])) {
        // LOGD("freeing maxDimensions");
        free(QNN_TENSOR_GET_DIMENSIONS(&tensors[tensorIdx]));
      }
      if (m_bufferAlloc == BufferAlloc::SHARED_BUFFER) {
        m_bufferManager->deregisterTensorFusedBuffer(&(tensors[tensorIdx]));
      } else {
        m_bufferManager->freeTensorBuffer(&(tensors[tensorIdx]));
      }
      m_freeTensorsPointerSet.insert(&(tensors[tensorIdx]));
    }
    free(tensors);
    tensors = nullptr;
  }

  return true;
}

// Clean up all tensors after execution.
bool IOTensor::tearDownTensors(std::vector<Qnn_Tensor_t*>& tensors, uint32_t numTensors) {
  for (Qnn_Tensor_t* tensor : tensors) {
    tearDownTensors(tensor, numTensors);
  }

  return true;
}

bool IOTensor::tearDownTensors(std::vector<Qnn_Tensor_t>& tensors) {
  return tearDownTensors(tensors.data(), tensors.size());
}

// Clean up all tensors after execution.
bool IOTensor::tearDownTensors(std::unordered_map<std::string, Qnn_Tensor_t*>& tensors,
                               std::unordered_map<std::string, uint32_t>& tensorCountMap) {
  for (auto& tensor : tensors) {
    tearDownTensors(tensor.second, tensorCountMap[tensor.first]);
  }

  return true;
}

// Clean up all tensors after execution.
bool IOTensor::tearDownTensors(std::vector<std::unordered_map<std::string, Qnn_Tensor_t*>>& tensors,
                               std::unordered_map<std::string, uint32_t>& tensorCountMap) {
  for (auto& tensor : tensors) {
    tearDownTensors(tensor, tensorCountMap);
  }

  return true;
}

bool IOTensor::deepCopyQnnTensorInfo(Qnn_Tensor_t* dest, Qnn_Tensor_t* src) {
  if (nullptr == dest || nullptr == src) {
    rwkvmobile::LOGE("Received nullptr");
    return false;
  }

  // set tensor.version before using QNN_TENSOR_SET macros, as they require the version to be set
  // to correctly assign values
  dest->version          = src->version;
  const char* tensorName = QNN_TENSOR_GET_NAME(src);
  if (!tensorName) {
    QNN_TENSOR_SET_NAME(dest, nullptr);
  } else {
    QNN_TENSOR_SET_NAME(dest, __strdup(tensorName));
  }
  QNN_TENSOR_SET_ID(dest, QNN_TENSOR_GET_ID(src));
  QNN_TENSOR_SET_TYPE(dest, QNN_TENSOR_GET_TYPE(src));
  QNN_TENSOR_SET_DATA_FORMAT(dest, QNN_TENSOR_GET_DATA_FORMAT(src));
  QNN_TENSOR_SET_DATA_TYPE(dest, QNN_TENSOR_GET_DATA_TYPE(src));
  Qnn_QuantizeParams_t qParams = QNN_QUANTIZE_PARAMS_INIT;
  qParams.encodingDefinition   = QNN_TENSOR_GET_QUANT_PARAMS(src).encodingDefinition;
  qParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
  if (QNN_TENSOR_GET_QUANT_PARAMS(src).quantizationEncoding ==
      QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
    qParams.quantizationEncoding = QNN_TENSOR_GET_QUANT_PARAMS(src).quantizationEncoding;
    qParams.scaleOffsetEncoding  = QNN_TENSOR_GET_QUANT_PARAMS(src).scaleOffsetEncoding;
  } else if (QNN_TENSOR_GET_QUANT_PARAMS(src).quantizationEncoding ==
             QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET) {
    qParams.quantizationEncoding = QNN_TENSOR_GET_QUANT_PARAMS(src).quantizationEncoding;
    qParams.axisScaleOffsetEncoding.axis =
        QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.axis;
    qParams.axisScaleOffsetEncoding.numScaleOffsets =
        QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.numScaleOffsets;
    if (QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.numScaleOffsets > 0) {
      qParams.axisScaleOffsetEncoding.scaleOffset = (Qnn_ScaleOffset_t*)malloc(
          QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.numScaleOffsets *
          sizeof(Qnn_ScaleOffset_t));
      if (qParams.axisScaleOffsetEncoding.scaleOffset) {
        for (size_t idx = 0;
             idx < QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.numScaleOffsets;
             idx++) {
          qParams.axisScaleOffsetEncoding.scaleOffset[idx].scale =
              QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.scaleOffset[idx].scale;
          qParams.axisScaleOffsetEncoding.scaleOffset[idx].offset =
              QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.scaleOffset[idx].offset;
        }
      }
    }
  }
  QNN_TENSOR_SET_QUANT_PARAMS(dest, qParams);
  QNN_TENSOR_SET_RANK(dest, QNN_TENSOR_GET_RANK(src));
  QNN_TENSOR_SET_DIMENSIONS(dest, nullptr);
  if (QNN_TENSOR_GET_RANK(src) > 0) {
    QNN_TENSOR_SET_DIMENSIONS(dest, (uint32_t*)malloc(QNN_TENSOR_GET_RANK(src) * sizeof(uint32_t)));
    if (QNN_TENSOR_GET_DIMENSIONS(dest)) {
      memcpy(QNN_TENSOR_GET_DIMENSIONS(dest),
             QNN_TENSOR_GET_DIMENSIONS(src),
             QNN_TENSOR_GET_RANK(src) * sizeof(uint32_t));
    }
  }

  return true;
}
