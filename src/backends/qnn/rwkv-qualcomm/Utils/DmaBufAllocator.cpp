//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <dlfcn.h>
#include <fcntl.h>
#include <linux/dma-buf.h>
#include <pthread.h>
#include <stdlib.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "DmaBufAllocator.hpp"
#include "QnnMem.h"
#include "QnnTypeMacros.hpp"

QnnDmaBufferAllocator::QnnDmaBufferAllocator(Qnn_ContextHandle_t contextHandle,
                                             QNN_INTERFACE_VER_TYPE* qnnInterface)
    : m_libDmaBufHeapHandle(nullptr),
      m_dmaBufCreate(nullptr),
      m_dmaBufAlloc(nullptr),
      m_dmaBufDeinit(nullptr),
      m_qnnInterface(qnnInterface),
      m_contextHandle(contextHandle) {}

bool QnnDmaBufferAllocator::initialize() {
  // On Android, 32-bit and 64-bit libdmaBufheap.so can be found at /system/lib and /system/lib64
  //  respectively.
  const std::string defaultLibPaths[] = {"libdmabufheap.so", "libdmabufheap.so.0"};
  for (const auto& path : defaultLibPaths) {
    m_libDmaBufHeapHandle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (m_libDmaBufHeapHandle != nullptr) {
      break;
    }
  }
  if (nullptr == m_libDmaBufHeapHandle) {
    rwkvmobile::LOGE("Unable to load backend. dlerror(): %s", dlerror());
    return false;
  }
  m_dmaBufCreate =
      (DmaBufCreateFn_t)dlsym(m_libDmaBufHeapHandle, "CreateDmabufHeapBufferAllocator");
  m_dmaBufAlloc  = (DmaBufAllocFn_t)dlsym(m_libDmaBufHeapHandle, "DmabufHeapAlloc");
  m_dmaBufDeinit = (DmaBufDeinitFn_t)dlsym(m_libDmaBufHeapHandle, "FreeDmabufHeapBufferAllocator");
  if (nullptr == m_dmaBufCreate || nullptr == m_dmaBufAlloc || nullptr == m_dmaBufDeinit) {
    rwkvmobile::LOGE("Unable to access symbols in libdmaBufheap. dlerror(): %s", dlerror());
    return false;
  }
  return true;
}

QnnDmaBufferAllocator::~QnnDmaBufferAllocator() {
  if (m_libDmaBufHeapHandle) {
    dlclose(m_libDmaBufHeapHandle);
    m_libDmaBufHeapHandle = nullptr;
  }
}

DmaBufferData* QnnDmaBufferAllocator::getDmaBufTensorData(Qnn_Tensor_t* tensor) {
  if (tensor == nullptr) return nullptr;
  Qnn_MemHandle_t mem_handle = QNN_TENSOR_GET_MEM_HANDLE(tensor);
  if (mem_handle == nullptr) return nullptr;
  return &m_memHandleToDmaBufMem.at(mem_handle);
}

void* QnnDmaBufferAllocator::getBuffer(Qnn_Tensor_t* tensor) {
  if (!tensor) {
    rwkvmobile::LOGW("DmaBufferAllocator: getBuffer: received a null pointer to a tensor");
    return nullptr;
  }
  if (m_tensorToDmaBufferData.find(tensor) == m_tensorToDmaBufferData.end()) {
    rwkvmobile::LOGE("DmaBufferAllocator: Tensor not found with address = %p", tensor);
    return nullptr;
  }
  DmaBufferData dmaBufferData = m_tensorToDmaBufferData[tensor];
  return dmaBufferData.memPointer;
}

int QnnDmaBufferAllocator::getFd(Qnn_Tensor_t* tensor) {
  DmaBufferData* data = getDmaBufTensorData(tensor);
  if (data == nullptr) {
    rwkvmobile::LOGE("DmaBufferAllocator: getFd : Couldn't find tensor %p", tensor);
    return -1;
  }
  return data->fd;
}

size_t QnnDmaBufferAllocator::getOffset(Qnn_Tensor_t* tensor) {
  DmaBufferData* data = getDmaBufTensorData(tensor);
  if (data == nullptr) {
    rwkvmobile::LOGE("DmaBufferAllocator: getOffset : Couldn't find tensor %p", tensor);
    return 0;
  }
  return data->offset;
}

size_t QnnDmaBufferAllocator::getBufferSize(Qnn_Tensor_t* tensor) {
  DmaBufferData* data = getDmaBufTensorData(tensor);
  if (data == nullptr) {
    rwkvmobile::LOGE("DmaBufferAllocator: getBufferSize : Couldn't find tensor %p", tensor);
    return 0;
  }
  return data->totalBufferSize;
};

size_t QnnDmaBufferAllocator::getTotalBufferSize(Qnn_Tensor_t* tensor) {
  DmaBufferData* data = getDmaBufTensorData(tensor);
  if (data == nullptr) {
    rwkvmobile::LOGE("DmaBufferAllocator: getTotalBufferSize : Couldn't find tensor %p", tensor);
    return 0;
  }
  return data->totalBufferSize;
}

bool QnnDmaBufferAllocator::allocateTensorBuffer(Qnn_Tensor_t* tensor, size_t tensorDataSize) {
  if (m_libDmaBufHeapHandle == nullptr) {
    rwkvmobile::LOGE("DmaBufferAllocator not initialized");
    return false;
  }

  if (!tensor) {
    rwkvmobile::LOGE("DmaBufferAllocator: Received nullptr for tensor");
    return false;
  }

  if (m_tensorToDmaBufferData.find(tensor) != m_tensorToDmaBufferData.end()) {
    rwkvmobile::LOGE("DmaBufferAllocator: Tensor already allocated");
    return false;
  }

  void* dmaBufferAllocator = m_dmaBufCreate();
  if (dmaBufferAllocator == nullptr) {
    rwkvmobile::LOGE("DmaBufferAllocator: nullptr returned for CreateDmabufHeapBufferAllocator().");
    return false;
  }

  int fd = m_dmaBufAlloc(dmaBufferAllocator, "qcom,system", tensorDataSize, 0, 0);
  if (fd < 0) {
    rwkvmobile::LOGE("DmaBufAlloc returned a invalid file descriptor = %d", fd);
    return false;
  }

  void* memPointer = mmap(nullptr, tensorDataSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (MAP_FAILED == memPointer) {
    printf("DmaBufferAllocator: Unable to open file returned by DmaBufAlloc with mmap");
    return false;
  }

  Qnn_MemDescriptor_t memDescriptor = {
      {QNN_TENSOR_GET_RANK(tensor), QNN_TENSOR_GET_DIMENSIONS(tensor), nullptr},
      QNN_TENSOR_GET_DATA_TYPE(tensor),
      QNN_MEM_TYPE_DMA_BUF,
      {.dmaBufInfo = {fd, memPointer}}};
  QNN_TENSOR_SET_MEM_TYPE(tensor, QNN_TENSORMEMTYPE_MEMHANDLE);
  QNN_TENSOR_SET_MEM_HANDLE(tensor, nullptr);
  Qnn_MemHandle_t memHandle = QNN_TENSOR_GET_MEM_HANDLE(tensor);

  if (QNN_SUCCESS !=
      m_qnnInterface->memRegister(m_contextHandle, &memDescriptor, 1, &(memHandle))) {
    rwkvmobile::LOGE("DmaBufferAllocator: Failure to register ion memory with the backend");
    return false;
  }
  rwkvmobile::LOGD(
      "DmaBufferAllocator: Memregister successful with handle %p for DMA buffer with size: %zu and "
      "fd %d",
      memHandle,
      tensorDataSize,
      fd);
  QNN_TENSOR_SET_MEM_HANDLE(tensor, memHandle);
  m_tensorToDmaBufferData.insert(
      {tensor, DmaBufferData(dmaBufferAllocator, fd, memPointer, tensorDataSize)});

  return true;
}

bool QnnDmaBufferAllocator::freeTensorBuffer(Qnn_Tensor_t* tensor) {
  if (!tensor) {
    rwkvmobile::LOGE("DmaBufferAllocator: Received nullptr for tensor");
    return false;
  }
  auto memHandle = QNN_TENSOR_GET_MEM_HANDLE(tensor);
  if (QNN_SUCCESS != m_qnnInterface->memDeRegister(&memHandle, 1)) {
    rwkvmobile::LOGE("DmaBufferAllocator: Failed to deregister custom memory handle with the backend");
    return false;
  }
  if (m_tensorToDmaBufferData.find(tensor) == m_tensorToDmaBufferData.end()) {
    rwkvmobile::LOGE("DmaBufferAllocator: Tensor not found with address = %p", tensor);
    return false;
  }
  DmaBufferData dmaBufferData = m_tensorToDmaBufferData[tensor];
  if (!m_dmaBufDeinit) {
    rwkvmobile::LOGE("DmaBufferAllocator: DmaBuf Deinit function pointer is null");
    return false;
  }
  munmap(dmaBufferData.memPointer, dmaBufferData.totalBufferSize);
  m_dmaBufDeinit(dmaBufferData.dmaBufferAllocator);
  m_tensorToDmaBufferData.erase(tensor);
  return true;
}

bool QnnDmaBufferAllocator::useSameMemory(Qnn_Tensor_t* dest, Qnn_Tensor_t* src) {
  if (nullptr == dest || nullptr == src) {
    rwkvmobile::LOGE("DmaBufferAllocator: Received nullptr");
    return false;
  }
  if (m_tensorToDmaBufferData.find(src) == m_tensorToDmaBufferData.end()) {
    rwkvmobile::LOGE("DmaBufferAllocator: Src Tensor not found");
    return false;
  }

  QNN_TENSOR_SET_MEM_TYPE(dest, QNN_TENSOR_GET_MEM_TYPE(src));
  QNN_TENSOR_SET_MEM_HANDLE(dest, QNN_TENSOR_GET_MEM_HANDLE(src));
  m_tensorToDmaBufferData.insert({dest, m_tensorToDmaBufferData[src]});
  m_sameMemoryFreeTensors.insert(dest);
  return true;
}

bool QnnDmaBufferAllocator::beforeWriteToBuffer(Qnn_Tensor_t* tensor) {
  if (!tensor) {
    rwkvmobile::LOGW("beforeWriteToBuffer: received a null pointer to a tensor");
    return false;
  }
  if (m_tensorToDmaBufferData.find(tensor) == m_tensorToDmaBufferData.end()) {
    rwkvmobile::LOGE("beforeWriteToBuffer: Tensor not found with address = %p", tensor);
    return false;
  }
  DmaBufferData dmaBufferData  = m_tensorToDmaBufferData[tensor];
  struct dma_buf_sync buf_sync = {};
  buf_sync.flags               = DMA_BUF_SYNC_START | DMA_BUF_SYNC_WRITE;
  auto ioctlReturnValue        = ioctl(dmaBufferData.fd, DMA_BUF_IOCTL_SYNC, &buf_sync);
  if (ioctlReturnValue) {
    rwkvmobile::LOGE(
        "beforeWriteToBuffer: Error preparing the cache for buffer writes."
        "The DMA_BUF_IOCTL_SYNC operation returned %d",
        ioctlReturnValue);
    return false;
  }
  return true;
}

bool QnnDmaBufferAllocator::afterWriteToBuffer(Qnn_Tensor_t* tensor) {
  if (!tensor) {
    rwkvmobile::LOGW("afterWriteToBuffer: received a null pointer to a tensor");
    return false;
  }
  if (m_tensorToDmaBufferData.find(tensor) == m_tensorToDmaBufferData.end()) {
    rwkvmobile::LOGE("afterWriteToBuffer: Tensor not found with address = %p", tensor);
    return false;
  }
  DmaBufferData dmaBufferData  = m_tensorToDmaBufferData[tensor];
  struct dma_buf_sync buf_sync = {};
  buf_sync.flags               = DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE;
  auto ioctlReturnValue        = ioctl(dmaBufferData.fd, DMA_BUF_IOCTL_SYNC, &buf_sync);
  if (ioctlReturnValue) {
    rwkvmobile::LOGE(
        "afterWriteToBuffer: Error close the cache after buffer writing."
        "The DMA_BUF_IOCTL_SYNC operation returned %d",
        ioctlReturnValue);
    return false;
  }
  return true;
}

bool QnnDmaBufferAllocator::beforeReadFromBuffer(Qnn_Tensor_t* tensor) {
  if (!tensor) {
    rwkvmobile::LOGW("beforeReadFromBuffer: received a null pointer to a tensor");
    return false;
  }
  if (m_tensorToDmaBufferData.find(tensor) == m_tensorToDmaBufferData.end()) {
    rwkvmobile::LOGE("beforeReadFromBuffer: Tensor not found with address = %p", tensor);
    return false;
  }
  DmaBufferData dmaBufferData  = m_tensorToDmaBufferData[tensor];
  struct dma_buf_sync buf_sync = {};
  buf_sync.flags               = DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ;
  auto ioctlReturnValue        = ioctl(dmaBufferData.fd, DMA_BUF_IOCTL_SYNC, &buf_sync);
  if (ioctlReturnValue) {
    rwkvmobile::LOGE(
        "beforeReadFromBuffer: Error preparing the cache for buffer reading."
        "The DMA_BUF_IOCTL_SYNC operation returned %d",
        ioctlReturnValue);
    return false;
  }
  return true;
}

bool QnnDmaBufferAllocator::afterReadFromBuffer(Qnn_Tensor_t* tensor) {
  if (!tensor) {
    rwkvmobile::LOGW("afterReadFromBuffer: received a null pointer to a tensor");
    return false;
  }
  if (m_tensorToDmaBufferData.find(tensor) == m_tensorToDmaBufferData.end()) {
    rwkvmobile::LOGE("afterReadFromBuffer: Tensor not found with address = %p", tensor);
    return false;
  }
  DmaBufferData dmaBufferData  = m_tensorToDmaBufferData[tensor];
  struct dma_buf_sync buf_sync = {};
  buf_sync.flags               = DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ;
  auto ioctlReturnValue        = ioctl(dmaBufferData.fd, DMA_BUF_IOCTL_SYNC, &buf_sync);
  if (ioctlReturnValue) {
    rwkvmobile::LOGE(
        "afterReadFromBuffer: Error closing the cache after buffer reading."
        "The DMA_BUF_IOCTL_SYNC operation returned %d",
        ioctlReturnValue);
    return false;
  }
  return true;
}
