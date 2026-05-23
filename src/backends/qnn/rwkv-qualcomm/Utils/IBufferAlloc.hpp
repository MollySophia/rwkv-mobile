//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "QnnTypes.h"

class IBufferAlloc {
 public:
  virtual ~IBufferAlloc() {}
  IBufferAlloc() {}
  virtual bool initialize()                                                                   = 0;
  virtual void* getBuffer(Qnn_Tensor_t* tensor)                                               = 0;
  virtual int getFd(Qnn_Tensor_t* tensor)                                                     = 0;
  virtual size_t getOffset(Qnn_Tensor_t* tensor)                                              = 0;
  virtual size_t getBufferSize(Qnn_Tensor_t* tensor)                                          = 0;
  virtual size_t getTotalBufferSize(Qnn_Tensor_t* tensor)                                     = 0;
  virtual bool allocateTensorBuffer(Qnn_Tensor_t* tensor, size_t tensorDataSize)              = 0;
  virtual bool allocateTensorBufferForContext(
      Qnn_Tensor_t* tensor, size_t tensorDataSize, Qnn_ContextHandle_t contextHandle) {
    (void)contextHandle;
    return allocateTensorBuffer(tensor, tensorDataSize);
  }
  virtual bool freeTensorBuffer(Qnn_Tensor_t* tensor)                                         = 0;
  virtual bool useSameMemory(Qnn_Tensor_t* dest, Qnn_Tensor_t* src)                           = 0;
  virtual bool useSameMemory(Qnn_Tensor_t* dest, Qnn_Tensor_t* src, int offset)               = 0;
  virtual bool useExternalMemory(Qnn_Tensor_t* dest, void* extMem)                            = 0;
  virtual void* allocateTensorFusedBuffer(uint64_t bufferSize, int32_t* fd)                   = 0;
  virtual bool allocateBuffers(const std::map<int, std::map<std::string, size_t>>& allocs_per_chunk,
                               std::map<std::string, std::pair<int, size_t>>& tensor_offsets) = 0;
  virtual bool mapFusedBufferOffset(Qnn_Tensor_t* tensor,
                                    size_t tensorDataSize,
                                    int32_t fd,
                                    uint32_t offset,
                                    uint64_t totalBufferSize,
                                    void* memPointer,
                                    Qnn_ContextHandle_t contextHandle)                        = 0;
  virtual bool mapFusedBufferOffset(
      Qnn_Tensor_t* tensor, int alloc_idx, size_t offset, Qnn_ContextHandle_t ctx, size_t size) = 0;

  virtual bool deregisterTensorFusedBuffer(Qnn_Tensor_t* tensor) = 0;
  virtual void freeFusedBuffers()                                = 0;

  // Functions to sync memory buffers for Read/Write using DmaBuf.
  virtual bool beforeWriteToBuffer(Qnn_Tensor_t* tensor) { return false; };
  virtual bool afterWriteToBuffer(Qnn_Tensor_t* tensor) { return false; };
  virtual bool beforeReadFromBuffer(Qnn_Tensor_t* tensor) { return false; };
  virtual bool afterReadFromBuffer(Qnn_Tensor_t* tensor) { return false; };
};
