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
template<typename T, int ELEMWIDTH, int DIMENSION>
class GPUArray;

template<typename T, int ELEMWIDTH>
class GPUArray<T, ELEMWIDTH, 1>
{
public:
  GPUArray() : data(nullptr), pitch(0) {}
  GPUArray(const GPUArray&& in)
  {
    data   = in[0];
    pitch  = in.getPitch();
    width  = in.getWidth();
    height = in.getHeight();
  }

  GPUArray& operator=(const GPUArray& in) = delete;
  ~GPUArray()
  {
    if (data != nullptr)
      cudaFree(data);
  }
  /** returns pointer to element(1D), row(2D), plane(3D)
   */
  T* operator[](int i);
  T*&& operator()(int i);
  void resize(int nBlocks, int nSplinesPerBlock);
  /// In Bytes
  size_t getWidth() const { return width; }
  /// In "Bytes"
  size_t getHeight() const { return height; }
  /// Actual width in bytes of allocated row
  size_t getPitch() const { return pitch; }
  void zero() { cudaMemset(data, 0, width); }

private:
  T* data;
  size_t pitch;
  size_t width;
  size_t height;
};

template<typename T, int ELEMWIDTH>
class GPUArray<T, ELEMWIDTH, 2>
{
public:
  GPUArray() : data(nullptr), pitch(0) {}
  GPUArray(const GPUArray& in)
  {
    data   = in[0];
    pitch  = in.getPitch();
    width  = in.getWidth();
    height = in.getHeight();
  }

  GPUArray& operator=(const GPUArray&& in) = delete;
  ~GPUArray()
  {
    if (data != nullptr)
      cudaFree(data);
  }
  /** returns pointer to element(1D), row(2D), plane(3D)
   */
  T* operator[](int i);
  T*&& operator()(int i);
  void resize(int nBlocks, int nSplinesPerBlock);
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

template<typename T, int ELEMWIDTH>
T* GPUArray<T, ELEMWIDTH, 1>::operator[](int i)
{
  return data != nullptr ? (T*)((char*)data + (i * sizeof(T))) : nullptr;
}

template<typename T, int ELEMWIDTH>
T* GPUArray<T, ELEMWIDTH, 2>::operator[](int i)
{
  return data != nullptr ? (T*)((char*)data + (i * pitch)) : nullptr;
}

template<typename T, int ELEMWIDTH>
T*&& GPUArray<T, ELEMWIDTH, 1>::operator()(int i)
{
  return data != nullptr ? (T*)((char*)data + (i * sizeof(T))) : nullptr;
}

template<typename T, int ELEMWIDTH>
T*&& GPUArray<T, ELEMWIDTH, 2>::operator()(int i)
{
  return data != nullptr ? (T*)((char*)data + (i * pitch)) : nullptr;
}

template<typename T, int ELEMWIDTH>
void GPUArray<T, ELEMWIDTH, 1>::resize(int nBlocks, int nSplinesPerBlock)
{
  int current_width = width;
  width              = sizeof(T) * nBlocks * ELEMWIDTH * nSplinesPerBlock;
  height             = 1;
  if(current_width < width)
  {
    if (data != nullptr)
      cudaFree(data);
    cudaError_t cu_err = cudaMalloc((void**)&data, width);
    if (cu_err != cudaError::cudaSuccess)
	throw std::runtime_error("Failed GPU allocation");
  }
}

template<typename T, int ELEMWIDTH>
void GPUArray<T, ELEMWIDTH, 2>::resize(int nBlocks, int nSplinesPerBlock)
{
  if (data != nullptr)
    cudaFree(data);
  width              = sizeof(T) * nBlocks * ELEMWIDTH;
  height             = sizeof(T) * nSplinesPerBlock;
  cudaError_t cu_err = cudaMallocPitch((void**)&data, &pitch, width, height);
  if (cu_err != cudaError::cudaSuccess)
    throw std::runtime_error("Failed GPU allocation");
}


} // namespace qmcplusplus

#endif
