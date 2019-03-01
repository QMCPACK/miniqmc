////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2019 QMCPACK developers.
//
// File developed by:
// Peter Doak, doakpw@ornl.gov, Oak Ridge National Lab
//
// File created by:
// Peter Doak, doakpw@ornl.gov, Oak Ridge National Lab
////////////////////////////////////////////////////////////////////////////////

#ifndef QMCPLUSPLUS_GPU_ARRAY_H
#define QMCPLUSPLUS_GPU_ARRAY_H

#include <stdexcept>
#include <cuda.h>
#include <cuda_runtime_api.h>

namespace qmcplusplus
{

/** The intention is that this be as simple as possible
 */
template<typename T, int ELEMWIDTH>
class GPUArray
{
public:
  GPUArray() : data(nullptr), pitch(0), data_owned(true) {}
  GPUArray(const GPUArray& in) : data_owned(false)
  {
    data = in[0];
    pitch = in.getPitch();
    width = in.getWidth();
    height = in.getHeight();
  }
  
  GPUArray& operator=(const GPUArray& in) = delete;
  ~GPUArray()
  {
    if(data != nullptr && data_owned)
      cudaFree(data);
  }
  T* operator[](int i) const { return data != nullptr ? (T*)((char*)data + (i * pitch)) : nullptr; }
  void resize(int nBlocks, int nSplinesPerBlock)
  {
    if(data_owned && data != nullptr)
      cudaFree(data);
    width = sizeof(T) * nBlocks * ELEMWIDTH;
    height = sizeof(T) * nSplinesPerBlock;
    cudaError_t cu_err = cudaMallocPitch((void**)&data,
					 &pitch,
					 width,
					 height);
    if(cu_err != cudaError::cudaSuccess)
      throw std::runtime_error("Failed GPU allocation");
  }
  /// In Bytes
  size_t getWidth() const { return width; }
  /// In "Bytes"
  size_t getHeight() const { return height; }
  /// Actual width in bytes of allocated row
  size_t getPitch() const { return pitch; }
  void zero() { cudaMemset2D(data, pitch, 0, width, height); }
private:
  bool data_owned;
  T* data;
  size_t pitch;
  size_t width;
  size_t height;
};

}

#endif