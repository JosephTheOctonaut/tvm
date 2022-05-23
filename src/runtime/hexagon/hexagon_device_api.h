/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#ifndef TVM_RUNTIME_HEXAGON_HEXAGON_DEVICE_API_H_
#define TVM_RUNTIME_HEXAGON_HEXAGON_DEVICE_API_H_

#include <tvm/runtime/device_api.h>

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "hexagon_buffer.h"
#include "hexagon_threadmanager.h"

namespace tvm {
namespace runtime {
namespace hexagon {

/*!
 * \brief Hexagon Device API that is compiled and run on Hexagon.
 */
class HexagonDeviceAPI final : public DeviceAPI {
 public:
  //! \brief Retrieve the global singleton instance of the HexagonDeviceAPI.
  static HexagonDeviceAPI* Global();

  //! \brief Constructor
  HexagonDeviceAPI() {
    #if defined(__hexagon__)
    thread_manager = new HexagonThreadManager(6, 16*(1<<10), 1<<10);
    thread_manager->GetStreamHandles(&free_streams);
    #endif
  }

  //! \brief Destructor
  ~HexagonDeviceAPI() {
    #if defined(__hexagon__)
    delete thread_manager;
    #endif
  }

  /*! \brief Currently unimplemented interface to specify the active
   *  Hexagon device.
   */
  void SetDevice(Device dev) final{};

  //! \brief Return the queried Hexagon device attribute.
  void GetAttr(Device dev, DeviceAttrKind kind, TVMRetValue* rv) final;

  //! \brief Currently unimplemented interface to synchronize a device stream.
  void StreamSync(Device dev, TVMStreamHandle stream) final {}

  //! \note Standard memory allocation methods of the DeviceAPI interface.
  //! \brief Allocate a flat allocation of global memory wrapped in a HexagonBuffer.
  void* AllocDataSpace(Device dev, size_t nbytes, size_t alignment, DLDataType type_hint) final;

  //! \brief Free the allocated HexagonBuffer.
  void FreeDataSpace(Device dev, void* ptr) final;

  /*! \brief Request a dynamically allocated HexagonBuffer from a workspace pool.
   *  \returns The underlying allocation pointer.
   */
  void* AllocWorkspace(Device dev, size_t size, DLDataType type_hint) final;

  //! Erase from tracked hexagon_buffer_map and free
  void FreeWorkspace(Device dev, void* data) final;

  /*!
   * \brief Allocate an Nd data space on device with memory scope support.
   *
   * If mem_scope is undefined or is "global", treat shape as the
   * tensor shape, to be flattened into an allocation of 1-d physical
   * memory.  This is done to maintain the semantics expected by callers of
   * DeviceAPI::AllocDataSpace, in cases where it has a valid return value.
   *
   * For other values of mem_scope, the shape is the N-d physical
   * shape of the allocation.
   *
   * \param dev The device to perform the operation.
   * \param ndim The number of dimensions of allocated tensor.
   * \param shape The shape of allocated tensor.
   * \param dtype The element type.
   * \param mem_scope The memory scope of the allocated tensor.
   * \return The allocated HexagonBuffer pointer.
   */
  void* AllocDataSpace(Device dev, int ndim, const int64_t* shape, DLDataType dtype,
                       Optional<String> mem_scope) final;

  /*!
   * \brief Allocate an Nd VTCM workspace.
   * \param dev The device to perform the operation.
   * \param ndim The number of dimensions of allocated tensor.
   * \param shape The shape of allocated tensor.
   * \param dtype The element type.
   * \return The allocated HexagonBuffer pointer.
   */
  void* AllocVtcmWorkspace(Device dev, int ndim, const int64_t* shape, DLDataType dtype,
                           Optional<String> mem_scope);

  //! \brief Free the allocated Nd VTCM workspace.
  void FreeVtcmWorkspace(Device dev, void* ptr);

  /*!
   * \brief Copy data from one storage to another.
   * \note This API is designed to support special memory with shape dependent layout.
   *       DLTensor's are passed with shape information to support these cases.
   * \param from The source array.
   * \param to The target array.
   * \param stream Optional stream object.
   */
  void CopyDataFromTo(DLTensor* from, DLTensor* to, TVMStreamHandle stream) final;

  // Interface functions for threads
  TVMStreamHandle CreateStream(Device dev);
  void FreeStream(Device dev, TVMStreamHandle stream);
  void SetStream(Device dev, TVMStreamHandle stream);
  void SyncStreamFromTo(Device dev, TVMStreamHandle event_src, TVMStreamHandle event_dst);
  void Dispatch(Device dev, PackedFunc f, TVMArgs args, TVMRetValue* rv, TVMStreamHandle stream = (void*)-1);
  void Start(Device dev);

 protected:
  //! Standard Device API interface to copy data from one storage to another.
  void CopyDataFromTo(const void* from, size_t from_offset, void* to, size_t to_offset, size_t size,
                      Device dev_from, Device dev_to, DLDataType type_hint,
                      TVMStreamHandle stream) final;

 private:
  /*! \brief Helper to allocate a HexagonBuffer and register the result
   *  in the owned buffer map.
   *  \return Raw data storage managed by the hexagon buffer
   */
  template <typename... Args>
  void* AllocateHexagonBuffer(Args&&... args) {
    auto buf = std::make_unique<HexagonBuffer>(std::forward<Args>(args)...);
    void* ptr = buf->GetPointer();
    hexagon_buffer_map_.insert({ptr, std::move(buf)});
    return ptr;
  }

  /*! \brief Helper to check if the device type is valid for the Hexagon Device API
   *  \return Boolean indicating whether the device type is valid
   */
  bool IsValidDevice(DLDevice dev) {
    // Added kDLCPU since we use hexagon as a sub-target of LLVM which by default maps to kDLCPU
    return (TVMDeviceExtType(dev.device_type) == kDLHexagon) ||
           (DLDeviceType(dev.device_type) == kDLCPU);
  }

  /*! \brief Helper to free a HexagonBuffer and unregister the result
   *  from the owned buffer map.
   */
  void FreeHexagonBuffer(void* ptr);
  //! Lookup table for the HexagonBuffer managing an allocation.
  std::unordered_map<void*, std::unique_ptr<HexagonBuffer>> hexagon_buffer_map_;

  #if defined(__hexagon__)
  HexagonThreadManager* thread_manager;
  #endif
  std::vector<TVMStreamHandle> free_streams;
  TVMStreamHandle active_stream = (void*)-1;
};
}  // namespace hexagon
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_HEXAGON_HEXAGON_DEVICE_API_H_
