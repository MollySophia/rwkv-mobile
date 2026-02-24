//==============================================================================
//
//  Copyright (c) Qualcomm Technorwkvmobile::LOGIes, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technorwkvmobile::LOGIes, Inc.
//
//==============================================================================

#include "HTP/QnnHtpMem.h"
#include "QnnMem.h"
#include "QnnTypeMacros.hpp"
#include "RpcMem.hpp"
#include "dlwrap.hpp"

#define RPCMEM_HEAP_ID_SYSTEM 25
#define RPCMEM_DEFAULT_FLAGS  1

#if 0
#define TRACE_MEMORY_ALLOC rwkvmobile::LOGI
#else
#define TRACE_MEMORY_ALLOC(fmt, ...)
#endif

RpcMem::RpcMem(Qnn_ContextHandle_t contextHandle, QNN_INTERFACE_VER_TYPE* qnnInterface)
    : m_libCdspRpc(nullptr),
      m_rpcMemAlloc(nullptr),
      m_rpcMemFree(nullptr),
      m_rpcMemToFd(nullptr),
      m_qnnInterface(qnnInterface),
      m_contextHandle(contextHandle) {
  (void)m_contextHandle;
}

bool RpcMem::initialize() {
  // On Android, 32-bit and 64-bit libcdsprpc.so can be found at /vendor/lib and /vendor/lib64
  // respectively. On Windows, it's installed into something like this
  //      c:\Windows\System32\DriverStore\FileRepository\qcnspmcdm8380.inf_arm64_30b9cc995571de6a\libcdsprpc.dll
#ifdef _WIN32
  const char* dsprpc_so = "libcdsprpc.dll";
#else
  const char* dsprpc_so = "libcdsprpc.so";
#endif

  m_libCdspRpc = dlopen(dsprpc_so, RTLD_NOW | RTLD_LOCAL);
  if (nullptr == m_libCdspRpc) {
    rwkvmobile::LOGE("Unable to load backend. dlerror(): %s", dlerror());
    return false;
  }
  m_rpcMemAlloc = (RpcMemAllocFn_t)dlsym(m_libCdspRpc, "rpcmem_alloc");
  m_rpcMemFree  = (RpcMemFreeFn_t)dlsym(m_libCdspRpc, "rpcmem_free");
  m_rpcMemToFd  = (RpcMemToFdFn_t)dlsym(m_libCdspRpc, "rpcmem_to_fd");
  if (nullptr == m_rpcMemAlloc || nullptr == m_rpcMemFree || nullptr == m_rpcMemToFd) {
    rwkvmobile::LOGE("Unable to access symbols in libcdsprpc. dlerror(): %s", dlerror());
    return false;
  }

  return true;
}

RpcMem::~RpcMem() {
  // TODO: Fix libcdsprpc.so handle closing order issue
  // if (m_libCdspRpc) {
  //   rwkvmobile::LOGI("Closing libcdsprpc.so handle");
  //   dlclose(m_libCdspRpc);
  // }
}

RpcMemTensorData* RpcMem::getRpcMemTensorData(Qnn_Tensor_t* tensor) {
  if (tensor == nullptr) return nullptr;
  Qnn_MemHandle_t mem_handle = QNN_TENSOR_GET_MEM_HANDLE(tensor);
  if (mem_handle == nullptr) return nullptr;
  // rwkvmobile::LOGI("RpcMem :: getRpcMemTensorData %s mem_handle=%p", QNN_TENSOR_GET_NAME(tensor), mem_handle);
  return &m_memHandleToRpcMem.at(mem_handle);
}

void* RpcMem::getBuffer(Qnn_Tensor_t* tensor) {
  RpcMemTensorData* data = getRpcMemTensorData(tensor);
  if (data == nullptr) {
    rwkvmobile::LOGE("getBuffer : Couldn't find tensor %p", tensor);
    return nullptr;
  }
  return data->memPointer;
}

int RpcMem::getFd(Qnn_Tensor_t* tensor) {
  RpcMemTensorData* data = getRpcMemTensorData(tensor);
  if (data == nullptr) {
    rwkvmobile::LOGE("getFd : Couldn't find tensor %p", tensor);
    return -1;
  }
  return data->fd;
}

size_t RpcMem::getOffset(Qnn_Tensor_t* tensor) {
  RpcMemTensorData* data = getRpcMemTensorData(tensor);
  if (data == nullptr) {
    rwkvmobile::LOGE("getOffset : Couldn't find tensor %p", tensor);
    return 0;
  }
  return data->offset;
}

size_t RpcMem::getBufferSize(Qnn_Tensor_t* tensor) {
  RpcMemTensorData* data = getRpcMemTensorData(tensor);
  if (data == nullptr) {
    rwkvmobile::LOGE("getBufferSize : Couldn't find tensor %p", tensor);
    return 0;
  }
  return data->size;
};

size_t RpcMem::getTotalBufferSize(Qnn_Tensor_t* tensor) {
  RpcMemTensorData* data = getRpcMemTensorData(tensor);
  if (data == nullptr) {
    rwkvmobile::LOGE("getTotalBufferSize : Couldn't find tensor %p", tensor);
    return 0;
  }
  return data->totalBufferSize;
}

bool RpcMem::allocateTensorBuffer(Qnn_Tensor_t* tensor, size_t tensorDataSize) {
  if (m_libCdspRpc == nullptr) {
    rwkvmobile::LOGE("RpcMem not initialized");
    return false;
  }
  if (!tensor) {
    rwkvmobile::LOGE("Received nullptr for tensor");
    return false;
  }
  if (m_tensorToRpcMem.find(tensor) != m_tensorToRpcMem.end()) {
    rwkvmobile::LOGE("Tensor already allocated");
    return false;
  }

  auto memPointer = m_rpcMemAlloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, tensorDataSize);
  auto status     = true;
  if (!memPointer) {
    rwkvmobile::LOGE("rpcmem_alloc failure");
    status = false;
  }
  int memfd = -1;
  if (status == true) {
    memfd = m_rpcMemToFd(memPointer);
    if (memfd == -1) {
      rwkvmobile::LOGE("rpcmem_to_fd failure");
      status = false;
    }
  }
  if (status == true) {
    Qnn_MemDescriptor_t memDescriptor = {
        {QNN_TENSOR_GET_RANK(tensor), QNN_TENSOR_GET_DIMENSIONS(tensor), nullptr},
        QNN_TENSOR_GET_DATA_TYPE(tensor),
        QNN_MEM_TYPE_ION,
        {{-1}}};
    memDescriptor.ionInfo.fd = memfd;
    QNN_TENSOR_SET_MEM_TYPE(tensor, QNN_TENSORMEMTYPE_MEMHANDLE);
    QNN_TENSOR_SET_MEM_HANDLE(tensor, nullptr);

    Qnn_MemHandle_t memHandle = QNN_TENSOR_GET_MEM_HANDLE(tensor);
    if (QNN_SUCCESS !=
        m_qnnInterface->memRegister(m_contextHandle, &memDescriptor, 1, &(memHandle))) {
      const char* tname = QNN_TENSOR_GET_NAME(tensor);
      rwkvmobile::LOGE("memRegister fail %s (ctx=%p fd=%d)", tname, m_contextHandle, memfd);
      status = false;
    }
    QNN_TENSOR_SET_MEM_HANDLE(tensor, memHandle);
    rwkvmobile::LOGD("RpcMem :: allocateTensorBuffer %s mem_handle=%p", QNN_TENSOR_GET_NAME(tensor), memHandle);
    m_memHandleToRpcMem.insert({memHandle, RpcMemTensorData(memfd, memPointer, tensorDataSize)});
  }
  if (status == true) {
    m_tensorToRpcMem.insert({tensor, RpcMemTensorData(memfd, memPointer, tensorDataSize)});
  }
  if (status == false) {
    if (m_rpcMemFree) {
      m_rpcMemFree(memPointer);
    }
  }
  return status;
}

bool RpcMem::freeTensorBuffer(Qnn_Tensor_t* tensor) {
  if (!tensor) {
    rwkvmobile::LOGE("Received nullptr for tensor");
    return false;
  }

  if (m_sameMemoryFreeTensors.find(tensor) != m_sameMemoryFreeTensors.end()) {
    if (m_tensorToRpcMem.find(tensor) == m_tensorToRpcMem.end()) {
      rwkvmobile::LOGE("Tensor not found");
      return false;
    }
    m_tensorToRpcMem.erase(tensor);
  } else {
    auto memHandle = QNN_TENSOR_GET_MEM_HANDLE(tensor);
    if (QNN_SUCCESS != m_qnnInterface->memDeRegister(&memHandle, 1)) {
      rwkvmobile::LOGE("Failed to deregister ion memory with the backend");
      return false;
    }
    QNN_TENSOR_SET_MEM_TYPE(tensor, QNN_TENSORMEMTYPE_UNDEFINED);
    if (m_tensorToRpcMem.find(tensor) == m_tensorToRpcMem.end()) {
      rwkvmobile::LOGE("Tensor not found");
      return false;
    }
    if (m_rpcMemFree) {
      m_rpcMemFree(m_tensorToRpcMem[tensor].memPointer);
    }
    m_tensorToRpcMem.erase(tensor);
  }

  return true;
}

bool RpcMem::useSameMemory(Qnn_Tensor_t* dest, Qnn_Tensor_t* src) {
  if (nullptr == dest || nullptr == src) {
    rwkvmobile::LOGE("Received nullptr");
    return false;
  }
  if (m_tensorToRpcMem.find(src) == m_tensorToRpcMem.end()) {
    rwkvmobile::LOGE("Src Tensor not found");
    return false;
  }

  // if (false == freeTensorBuffer(dest)) {
  //   rwkvmobile::LOGE("Failed to free dest tensor");
  //   return false;
  // }

  QNN_TENSOR_SET_MEM_TYPE(dest, QNN_TENSOR_GET_MEM_TYPE(src));
  QNN_TENSOR_SET_MEM_HANDLE(dest, QNN_TENSOR_GET_MEM_HANDLE(src));
  m_tensorToRpcMem.insert({dest, m_tensorToRpcMem[src]});
  m_sameMemoryFreeTensors.insert(dest);

  return true;
}

bool RpcMem::useSameMemory(Qnn_Tensor_t* dest, Qnn_Tensor_t* src, int offset) {
  if (nullptr == dest || nullptr == src) {
    rwkvmobile::LOGE("Received nullptr");
    return false;
  }
  if (m_tensorToRpcMem.find(src) == m_tensorToRpcMem.end()) {
    rwkvmobile::LOGE("Src Tensor not found");
    return false;
  }

  if (false == freeTensorBuffer(dest)) {
    return false;
  }

  QNN_TENSOR_SET_MEM_TYPE(dest, QNN_TENSOR_GET_MEM_TYPE(src));
  QNN_TENSOR_SET_MEM_HANDLE(dest, QNN_TENSOR_GET_MEM_HANDLE(src));
  m_tensorToRpcMem.insert({dest, m_tensorToRpcMem[src]});
  m_sameMemoryFreeTensors.insert(dest);

  return true;
}

bool RpcMem::useExternalMemory(Qnn_Tensor_t* dest, void* extMem) {
  rwkvmobile::LOGE("We don't support external memory feature for shared buffers yet!");
  return false;
}

void* RpcMem::allocateTensorFusedBuffer(uint64_t bufferSize, int32_t* fd) {
  *fd = -1;
  if (m_libCdspRpc == nullptr) {
    rwkvmobile::LOGE("RpcMem not initialized for fused buffer");
    return nullptr;
  }

  void* memPointer = m_rpcMemAlloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, bufferSize);
  if (!memPointer) {
    rwkvmobile::LOGE("Not able to allocate fused buffer of size: %lu", (unsigned long)bufferSize);
    return nullptr;
  }

  m_fusedBuffers.push_back({memPointer, bufferSize});
  rwkvmobile::LOGD("Successfully allocated fused buffer at %p with size %lu",
            memPointer,
            (unsigned long)bufferSize);

  if ((*fd = m_rpcMemToFd(memPointer)) == -1) {
    rwkvmobile::LOGE("Not able to get fd for the fused buffer of size: %lu", (unsigned long)bufferSize);
    return nullptr;
  }

  rwkvmobile::LOGD("Retrieved fd %d for pointer %p", *fd, memPointer);
  return memPointer;
}

bool RpcMem::allocateBuffers(const std::map<int, std::map<std::string, size_t>>& allocs_per_chunk,
                             std::map<std::string, std::pair<int, size_t>>& tensor_offsets) {
  int alloc_chunk_idx     = m_fusedBuffers.size();
  int num_alloc_chunks    = 0;
  size_t total_alloc_size = 0;

  for (auto& [_, tensor_sizes] : allocs_per_chunk) {
    // Calculate total allocation chunk size
    size_t alloc_chunk_size = 0;
    for (const auto& [tensor_name, tensor_size] : tensor_sizes) {
      tensor_offsets[tensor_name] = {alloc_chunk_idx, alloc_chunk_size};
      alloc_chunk_size += tensor_size;
    }

    // Allocate chunk for this unique context set
    if (alloc_chunk_size <= 0) {
      rwkvmobile::LOGE("Unexpected chunk size detected. Please re-check IO allocations");
      return false;
    }

    m_fusedFds.push_back(0);
    if (!allocateTensorFusedBuffer(alloc_chunk_size, &m_fusedFds.back()))  //
      return false;
    total_alloc_size += alloc_chunk_size;
    alloc_chunk_idx++;
    num_alloc_chunks++;
  }
  rwkvmobile::LOGI("Allocated total size = %lu across %d buffers",
           (unsigned long)total_alloc_size,
           num_alloc_chunks);
  return true;
}

bool RpcMem::mapFusedBufferOffset(Qnn_Tensor_t* tensor,
                                  size_t tensorDataSize,
                                  int32_t fd,
                                  uint32_t offset,
                                  uint64_t totalBufferSize,
                                  void* memPointer,
                                  Qnn_ContextHandle_t contextHandle) {
  if (m_libCdspRpc == nullptr) {
    rwkvmobile::LOGE("RpcMem not initialized");
    return false;
  }
  if (!tensor) {
    rwkvmobile::LOGE("Received nullptr for tensor");
    return false;
  }

  Qnn_ErrorHandle_t ret;
  const char* tname = QNN_TENSOR_GET_NAME(tensor);

  // Check if tensor already has a memHandle assigned
  Qnn_MemHandle_t cur_mem_handle = QNN_TENSOR_GET_MEM_HANDLE(tensor);
  if (cur_mem_handle != nullptr) {
    // Check if memHandle is already identical to requested buffer and offset
    RpcMemTensorData& cur_rpc_mem_data = m_memHandleToRpcMem.at(cur_mem_handle);
    if (cur_rpc_mem_data.fd == fd && cur_rpc_mem_data.offset == offset) {
      return true;
    }

    // updated offset, deregister previous mem_handle
    if (tensorDataSize == 0) tensorDataSize = cur_rpc_mem_data.size;
    TRACE_MEMORY_ALLOC("memDeRegister %-20s (fd=%d offset=%lu) memHandle=%p",
                       tname,
                       cur_rpc_mem_data.fd,
                       cur_rpc_mem_data.offset,
                       cur_mem_handle);
    m_memHandleToRpcMem.erase(cur_mem_handle);
    if ((ret = m_qnnInterface->memDeRegister(&cur_mem_handle, 1)) != QNN_SUCCESS) {
      rwkvmobile::LOGE(
          "memDeRegister ERROR(%lu) - %s memHandle=%p", (unsigned long)ret, tname, cur_mem_handle);
      return false;
    }
  } else {
    // For inital tensors, we need to check if the tensor can re-use a memHandle
    // from another tensor in the same context
    auto memConfig = std::make_tuple(fd, offset, contextHandle);
    if (memConfigList.find(memConfig) != memConfigList.end()) {
      auto& parentTensor              = memConfigList[memConfig];
      Qnn_MemHandle_t parentMemHandle = QNN_TENSOR_GET_MEM_HANDLE(parentTensor);
      QNN_TENSOR_SET_MEM_TYPE(tensor, QNN_TENSORMEMTYPE_MEMHANDLE);
      QNN_TENSOR_SET_MEM_HANDLE(tensor, parentMemHandle);
      TRACE_MEMORY_ALLOC("%-20s : Mapping to memHandle %p", tname, parentMemHandle);
      return true;
    }
  }

  // Register a new memHandle based on function arguments
  QnnMemHtp_Descriptor_t htp_mem_desciptor    = {QNN_HTP_MEM_SHARED_BUFFER, totalBufferSize, {0}};
  htp_mem_desciptor.sharedBufferConfig.fd     = fd;
  htp_mem_desciptor.sharedBufferConfig.offset = offset;

  Qnn_MemDescriptor_t mem_descriptor = {
      {QNN_TENSOR_GET_RANK(tensor), QNN_TENSOR_GET_DIMENSIONS(tensor), nullptr},
      QNN_TENSOR_GET_DATA_TYPE(tensor),
      QNN_MEM_TYPE_CUSTOM,
      {{-1}}};
  mem_descriptor.customInfo = &htp_mem_desciptor;

  Qnn_MemHandle_t mem_handle = nullptr;
  ret = m_qnnInterface->memRegister(contextHandle, &mem_descriptor, 1, &mem_handle);
  if (ret != QNN_SUCCESS) {
    rwkvmobile::LOGE("%-20s (ctx=%p fd=%d offset=%u)", tname, contextHandle, fd, offset);
    rwkvmobile::LOGE("memRegister ERROR(%lu)", (unsigned long)ret);
    return false;
  }

  TRACE_MEMORY_ALLOC("%-20s (ctx=%p fd=%d offset=%u) memPointer=%p memHandle=%p",
                     tname,
                     contextHandle,
                     fd,
                     offset,
                     ((uint8_t*)memPointer) + offset,
                     mem_handle);
  m_memHandleToRpcMem[mem_handle] = RpcMemTensorData(
      fd, ((uint8_t*)memPointer) + offset, tensorDataSize, totalBufferSize, offset);

  QNN_TENSOR_SET_MEM_TYPE(tensor, QNN_TENSORMEMTYPE_MEMHANDLE);
  QNN_TENSOR_SET_MEM_HANDLE(tensor, mem_handle);
  if (cur_mem_handle == nullptr)  // Cache memory config for initial memRegisters only
    memConfigList[std::make_tuple(fd, offset, contextHandle)] = tensor;
  m_tensorToRpcMem.insert({tensor, RpcMemTensorData(fd, (uint8_t*)memPointer + offset, tensorDataSize)});
  // m_tensorToRpcMem[tensor] = RpcMemTensorData(fd, ((uint8_t*)memPointer) + offset, tensorDataSize, totalBufferSize, offset);
  return true;
}

bool RpcMem::mapFusedBufferOffset(
    Qnn_Tensor_t* tensor, int alloc_idx, size_t offset, Qnn_ContextHandle_t ctx, size_t size) {
  return mapFusedBufferOffset(tensor,
                              size,
                              m_fusedFds[alloc_idx],
                              offset,
                              m_fusedBuffers[alloc_idx].second,
                              m_fusedBuffers[alloc_idx].first,
                              ctx);
}

bool RpcMem::deregisterTensorFusedBuffer(Qnn_Tensor_t* tensor) {
  if (!tensor) {
    rwkvmobile::LOGE("Received nullptr for tensor");
    return false;
  }

  if (m_tensorToRpcMem.find(tensor) == m_tensorToRpcMem.end()) {
    rwkvmobile::LOGE("Tensor not found");
    return false;
  }

  // We are not freeing memhandles here since they are already freed when
  // freeContext() gets called in the destructor of QnnApi class which
  // happens before this point

  // Qnn_MemHandle_t memHandle = QNN_TENSOR_GET_MEM_HANDLE(tensor);
  // rwkvmobile::LOGE("Interface handle %p memhandle %p", m_qnnInterface, memHandle);
  // if (QNN_SUCCESS != m_qnnInterface->memDeRegister(&memHandle, 1)) {
  //   rwkvmobile::LOGE("Failed to deregister ion memory with the backend");
  //   return false;
  // }

  QNN_TENSOR_SET_MEM_TYPE(tensor, QNN_TENSORMEMTYPE_UNDEFINED);
  QNN_TENSOR_SET_MEM_HANDLE(tensor, nullptr);
  m_tensorToRpcMem.erase(tensor);
  return true;
}

void RpcMem::freeFusedBuffers() {
  // for (auto& memHandle : m_orphanedMemHandles) {
  //   if (QNN_SUCCESS != m_qnnInterface->memDeRegister(&memHandle, 1)) {
  //     rwkvmobile::LOGE("Failed to deregister ion memory with the backend");
  //   }
  // }

  for (auto& [mem_ptr, buffer_size] : m_fusedBuffers) {
    rwkvmobile::LOGD("Freeing fused buffer %p (size=%lu)", mem_ptr, buffer_size);
    m_rpcMemFree(mem_ptr);
  }
}
