//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include <unordered_set>

#include "IBufferAlloc.hpp"
#include "logger.h"
#include "QnnInterface.h"

typedef void* (*RpcMemAllocFn_t)(int, uint32_t, int);
typedef void (*RpcMemFreeFn_t)(void*);
typedef int (*RpcMemToFdFn_t)(void*);

struct RpcMemTensorData {
  int fd;
  void* memPointer;
  size_t size;
  size_t totalBufferSize;
  size_t offset;
  RpcMemTensorData() : fd(-1), memPointer(nullptr), size(0), totalBufferSize(0), offset(0) {}
  RpcMemTensorData(int fdIn, void* memPointerIn, size_t sizeIn)
      : fd(fdIn), memPointer(memPointerIn), size(sizeIn), totalBufferSize(sizeIn), offset(0) {}
  RpcMemTensorData(
      int fdIn, void* memPointerIn, size_t sizeIn, size_t totalBufferSizeIn, size_t offsetIn)
      : fd(fdIn),
        memPointer(memPointerIn),
        size(sizeIn),
        totalBufferSize(totalBufferSizeIn),
        offset(offsetIn) {}
};

class RpcMem final : public IBufferAlloc {
 public:
  RpcMem(Qnn_ContextHandle_t contextHandle, QNN_INTERFACE_VER_TYPE* qnnInterface);
  // Disable copy constructors, r-value referencing, etc
  RpcMem(const RpcMem&)            = delete;
  RpcMem& operator=(const RpcMem&) = delete;
  RpcMem(RpcMem&&)                 = delete;
  RpcMem& operator=(RpcMem&&)      = delete;
  bool initialize() override;
  void* getBuffer(Qnn_Tensor_t* tensor) override;
  int getFd(Qnn_Tensor_t* tensor) override;

  size_t getOffset(Qnn_Tensor_t* tensor) override;

  size_t getBufferSize(Qnn_Tensor_t* tensor) override;

  size_t getTotalBufferSize(Qnn_Tensor_t* tensor) override;

  bool allocateTensorBuffer(Qnn_Tensor_t* tensor, size_t tensorDataSize) override;
  bool allocateTensorBufferForContext(
      Qnn_Tensor_t* tensor, size_t tensorDataSize, Qnn_ContextHandle_t contextHandle) override;

  bool freeTensorBuffer(Qnn_Tensor_t* tensor) override;
  bool useSameMemory(Qnn_Tensor_t* dest, Qnn_Tensor_t* src) override;
  bool useSameMemory(Qnn_Tensor_t* dest, Qnn_Tensor_t* src, int offset) override;

  bool useExternalMemory(Qnn_Tensor_t* dest, void* extMem) override;

  void* allocateTensorFusedBuffer(uint64_t bufferSize, int32_t* fd) override;
  bool allocateBuffers(const std::map<int, std::map<std::string, size_t>>& allocs_per_chunk,
                       std::map<std::string, std::pair<int, size_t>>& tensor_offsets) override;

  bool mapFusedBufferOffset(Qnn_Tensor_t* tensor,
                            size_t tensorDataSize,
                            int32_t fd,
                            uint32_t offset,
                            uint64_t totalBufferSize,
                            void* memPointer,
                            Qnn_ContextHandle_t contextHandle) override;
  bool deregisterTensorFusedBuffer(Qnn_Tensor_t* tensor) override;
  void freeFusedBuffers() override;
  bool mapFusedBufferOffset(Qnn_Tensor_t* tensor,
                            int alloc_idx,
                            size_t offset,
                            Qnn_ContextHandle_t ctx,
                            size_t size) override;
  virtual ~RpcMem();

 private:
  RpcMemTensorData* getRpcMemTensorData(Qnn_Tensor_t* tensor);

  // Pointer to the dlopen'd libcdsprpc.so shared library which contains
  // rpcmem_alloc, rpcmem_free, rpcmem_to_fd APIs
  void* m_libCdspRpc;
  // Function pointer to rpcmem_alloc
  RpcMemAllocFn_t m_rpcMemAlloc;
  // Function pointer to rpcmem_free
  RpcMemFreeFn_t m_rpcMemFree;
  // Function pointer to rpcmem_to_fd
  RpcMemToFdFn_t m_rpcMemToFd;
  QNN_INTERFACE_VER_TYPE* m_qnnInterface;
  Qnn_ContextHandle_t m_contextHandle;

  std::unordered_map<Qnn_Tensor_t*, RpcMemTensorData> m_tensorToRpcMem;
  std::unordered_set<Qnn_Tensor_t*> m_sameMemoryFreeTensors;
  std::vector<std::pair<void*, size_t>> m_fusedBuffers;  // vector<<memPointer, bufferSize>>
  std::vector<int32_t> m_fusedFds;
  std::unordered_set<Qnn_MemHandle_t> m_orphanedMemHandles;
  std::unordered_map<Qnn_MemHandle_t, RpcMemTensorData> m_memHandleToRpcMem;
  std::map<std::tuple<int, size_t, Qnn_ContextHandle_t>, Qnn_Tensor_t*> memConfigList;
};
