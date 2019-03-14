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

#include <utility>
#include <stdexcept>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "Numerics/Containers.h"


namespace qmcplusplus
{
struct CopyHome
{
  char* buffer;
  int width_;
  CopyHome(int width)
  {
    width_ = width;
    buffer = new char[width_];
  }
  void* operator()(void* data)
  {
    cudaError_t cu_err = cudaMemcpy(buffer, data, width_, cudaMemcpyDeviceToHost);
    if (cu_err != cudaError::cudaSuccess)
      throw std::runtime_error("Failed GPU to Host copy");
    return buffer;
  }
  ~CopyHome() { delete[] buffer; }
};

/** The intention is that this be as simple as possible
 *  However in the course of pursuing the solution to the * and ** problems for CUDA
 *  Things have gotten a bit disgusting. Needs to be fixed
 */
template<typename T, int ELEMWIDTH, int DIMENSION>
class GPUArray;

template<typename T, int ELEMWIDTH>
class GPUArray<T, ELEMWIDTH, 1>
{
public:
  GPUArray() : data(nullptr), pitch(0), width(0), height(0) {}
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
  /// We need these because we need to be able to copy back into CPU structures
  int getBlocks() const { return nBlocks_; }
  int getSplinesPerBlock() const { return nSplinesPerBlock_; }
  /// Actual width in bytes of allocated row
  size_t getPitch() const { return pitch; }
  void zero() { cudaMemset(data, 0, width); }
  T* get_devptr() { return data; }
  void pull(aligned_vector<T>& aVec)
  {
    CopyHome copy_home(width);
    T* buffer    = static_cast<T*>(copy_home(data));
    int elements = width / sizeof(T);
    aVec.resize(elements);
    for (int i = 0; i < elements; ++i)
    {
      aVec[i] = buffer[i];
    }
  }

  void pull(aligned_vector<aligned_vector<T>>& aVec)
  {
    CopyHome copy_home(width);
    T* buffer    = static_cast<T*>(copy_home(data));
    int elements = width / sizeof(T);
    for (int i = 0; i < nBlocks_; ++i)
    {
      for (int j = 0; j < nSplinesPerBlock_; ++j)
      {
        aVec[i][j] = buffer[i * nSplinesPerBlock_ + j];
      }
    }
  }

  void pull(VectorSoAContainer<T, ELEMWIDTH>& vSoA)
  {
    CopyHome copy_home(width);
    T* buffer = static_cast<T*>(copy_home(data));
    int elements = width / (ELEMWIDTH * sizeof(T));
    vSoA.resize(elements);
    // The data should now be in VSoAOrder
    std::memcpy(vSoA.data(), buffer, width);
    // for (int i = 0; i < elements; ++i)
    // {
    //   vSoA.data()
    //   // The Accessor thing in VSoA seems broken
    //   // TinyVector<T, ELEMWIDTH> tempTV(static_cast<const T* restrict>(buffer+i*ELEMWIDTH), 1);
    //   // vSoA(i) = tempTV;
    // } //
  }

  void pull(aligned_vector<VectorSoAContainer<T, ELEMWIDTH>>& av_vSoA)
  {
    CopyHome copy_home(width);
    void* buffer = copy_home(data);
    int elements = width / (ELEMWIDTH * sizeof(T));
    av_vSoA.resize(elements);
    for (int i = 0; i < nBlocks_; ++i)
    {
      for (int j = 0; j < nSplinesPerBlock_; j++)
	{
	  TinyVector<T, ELEMWIDTH> tempTV(static_cast<const T* restrict>(buffer), i * ELEMWIDTH);
	  av_vSoA[i](j) = tempTV;
	}
    }
  }


private:
  T* data;
  int nBlocks_;
  int nSplinesPerBlock_;
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
  nBlocks_         = nBlocks;
  nSplinesPerBlock_ = nSplinesPerBlock;

  int current_width = width;
  width             = sizeof(T) * nBlocks * ELEMWIDTH * nSplinesPerBlock;
  height            = 1;
  if (current_width < width)
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
