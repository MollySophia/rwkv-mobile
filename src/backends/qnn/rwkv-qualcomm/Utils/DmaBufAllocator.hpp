//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================
#pragma once

#include <map>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "IBufferAlloc.hpp"
#include "logger.h"
#include "QnnInterface.h"

typedef void* (*DmaBufCreateFn_t)();
typedef int (*DmaBufAllocFn_t)(void*, const char*, size_t, unsigned int, size_t);
typedef void (*DmaBufDeinitFn_t)(void*);

struct DmaBufferData {
  void* dmaBufferAllocator;
  int fd;
  void* memPointer;
  size_t totalBufferSize;
  int offset{0};
  DmaBufferData() : dmaBufferAllocator(nullptr), fd(-1), memPointer(nullptr), totalBufferSize(0) {}
  DmaBufferData(void* bufferAllocator, int fdIn, void* memPointerIn, size_t sizeIn)
      : dmaBufferAllocator(bufferAllocator),
        fd(fdIn),
        memPointer(memPointerIn),
        totalBufferSize(sizeIn) {}
};

class QnnDmaBufferAllocator final : public IBufferAlloc {
 public:
  QnnDmaBufferAllocator(Qnn_ContextHandle_t contextHandle, QNN_INTERFACE_VER_TYPE* qnnInterface);
  // Disable copy constructors, r-value referencing, etc
  QnnDmaBufferAllocator(const QnnDmaBufferAllocator&)            = delete;
  QnnDmaBufferAllocator& operator=(const QnnDmaBufferAllocator&) = delete;
  QnnDmaBufferAllocator(QnnDmaBufferAllocator&&)                 = delete;
  QnnDmaBufferAllocator& operator=(QnnDmaBufferAllocator&&)      = delete;

  bool initialize() override;
  void* getBuffer(Qnn_Tensor_t* tensor) override;
  int getFd(Qnn_Tensor_t* tensor) override;
  size_t getOffset(Qnn_Tensor_t* tensor) override;
  size_t getBufferSize(Qnn_Tensor_t* tensor) override;
  size_t getTotalBufferSize(Qnn_Tensor_t* tensor) override;

  bool freeTensorBuffer(Qnn_Tensor_t* tensor) override;

  bool allocateTensorBuffer(Qnn_Tensor_t* tensor, size_t tensorDataSize) override;
  bool useSameMemory(Qnn_Tensor_t* dest, Qnn_Tensor_t* src) override;

  virtual ~QnnDmaBufferAllocator();

  bool beforeWriteToBuffer(Qnn_Tensor_t* tensor) override;
  bool afterWriteToBuffer(Qnn_Tensor_t* tensor) override;
  bool beforeReadFromBuffer(Qnn_Tensor_t* tensor) override;
  bool afterReadFromBuffer(Qnn_Tensor_t* tensor) override;

  bool useSameMemory(Qnn_Tensor_t* dest, Qnn_Tensor_t* src, int offset) override {
    rwkvmobile::LOGW("Offset based tensors not supported!!");
    return false;
    ;
  }
  bool useExternalMemory(Qnn_Tensor_t* dest, void* extMem) override {
    rwkvmobile::LOGW("External Memory not supported!!");
    return false;
    ;
  }
  void* allocateTensorFusedBuffer(uint64_t bufferSize, int32_t* fd) override {
    rwkvmobile::LOGW("Fused Buffers not supported\n");
    return nullptr;
  };
  bool allocateBuffers(const std::map<int, std::map<std::string, size_t>>& allocs_per_chunk,
                       std::map<std::string, std::pair<int, size_t>>& tensor_offsets) override {
    rwkvmobile::LOGW("Fused Buffers not supported\n");
    return false;
  };
  bool mapFusedBufferOffset(Qnn_Tensor_t* tensor,
                            size_t tensorDataSize,
                            int32_t fd,
                            uint32_t offset,
                            uint64_t totalBufferSize,
                            void* memPointer,
                            Qnn_ContextHandle_t contextHandle) override {
    rwkvmobile::LOGW("Fused Buffers not supported\n");
    return false;
  };
  bool deregisterTensorFusedBuffer(Qnn_Tensor_t* tensor) override {
    rwkvmobile::LOGW("Fused Buffers not supported\n");
    return false;
  };
  void freeFusedBuffers() override { return; };
  bool mapFusedBufferOffset(Qnn_Tensor_t* tensor,
                            int alloc_idx,
                            size_t offset,
                            Qnn_ContextHandle_t ctx,
                            size_t size) override {
    rwkvmobile::LOGW("Fused Buffers not supported\n");
    return false;
  };

 private:
  DmaBufferData* getDmaBufTensorData(Qnn_Tensor_t* tensor);

  // Pointer to the dlopen'd libdmabufheap.so shared library which contains
  // dmaBufCreate, dmaBufAlloc, dmaBufDeinit
  void* m_libDmaBufHeapHandle;
  DmaBufCreateFn_t m_dmaBufCreate;
  DmaBufAllocFn_t m_dmaBufAlloc;
  DmaBufDeinitFn_t m_dmaBufDeinit;

  QNN_INTERFACE_VER_TYPE* m_qnnInterface;
  Qnn_ContextHandle_t m_contextHandle;

  std::unordered_map<Qnn_Tensor_t*, DmaBufferData> m_tensorToDmaBufferData;
  std::unordered_set<Qnn_Tensor_t*> m_sameMemoryFreeTensors;
  std::unordered_map<Qnn_MemHandle_t, DmaBufferData> m_memHandleToDmaBufMem;
};
