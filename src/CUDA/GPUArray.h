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
    cudaError_t cu_err = cudaMemcpy(buffer, data, width_, cudaMemcpyDefault);
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
  GPUArray() : data(nullptr), pitch(0), max_width_(0), width(0), height(0), owner(true) {}
  GPUArray(const GPUArray&& in)
  {
    owner = in.owner;
    data   = in[0];
    pitch  = in.getPitch();
    width  = in.getWidth();
    height = in.getHeight();
  }

  /** This lets GPUArray map an arbitrary cuda memory block.
   */
  GPUArray(T* dev_ptr, size_t size, int blocks, int splines_per_block)
  {
      owner = false;
      nBlocks_ = blocks;
      nSplinesPerBlock_ = splines_per_block;
      data = dev_ptr;
      width = size;
  }
  
  GPUArray& operator=(const GPUArray& in) = delete;
  ~GPUArray()
  {
      if (data != nullptr && owner == true)
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
    std::memcpy(aVec.data(), buffer, width);
  }

  /** get single value data back from the GPU
   */
  void pull(aligned_vector<aligned_vector<T>>& av_aVec, int block_offset = 0, int blocks = 0)
  {
    // This should in the default case be the entire width
    if(blocks == 0)
      blocks = nBlocks_;
    size_t this_width = std::min(blocks * nSplinesPerBlock_ * sizeof(T),
				 width - block_offset * nSplinesPerBlock_ * sizeof(T))  ;
    CopyHome copy_home(this_width);
    //The rside is operations on a double* so no sizeof multiplier
    void* offset_data = data + block_offset * nSplinesPerBlock_;
    T* buffer    = static_cast<T*>(copy_home(offset_data));
    int total_elements = this_width / sizeof(T);
    int this_blocks = std::min(nBlocks_, blocks);
    av_aVec.resize(this_blocks);
    for (int i = 0; i < this_blocks; ++i)
    {
      int elements = std::min(nSplinesPerBlock_, total_elements - i * nSplinesPerBlock_);
      aligned_vector<T>& aVec = av_aVec[i];
      aVec.resize(elements);
      assert((i * nSplinesPerBlock_ + elements)*sizeof(T) <= width);
      std::memcpy(aVec.data(), (void*)(buffer + i * nSplinesPerBlock_), elements * sizeof(T));
    }
  }

  void pull(VectorSoAContainer<T, ELEMWIDTH>& vSoA)
  {
    CopyHome copy_home(width);
    T* buffer = static_cast<T*>(copy_home(data));
    int elements = width / (ELEMWIDTH * sizeof(T));
    vSoA.resize(elements);
    std::memcpy(vSoA.data(), buffer, width);
  }

  void pull(aligned_vector<VectorSoAContainer<T, ELEMWIDTH>>& av_vSoA, int block_offset = 0, int blocks = 0)
  {
    if (blocks == 0)
      blocks = nBlocks_;
    size_t this_width = std::min(blocks * nSplinesPerBlock_ * ELEMWIDTH * sizeof(T),
				 width - block_offset * nSplinesPerBlock_ * ELEMWIDTH *sizeof(T))  ;
    
    CopyHome copy_home(this_width);
    void* offset_data = data + block_offset * nSplinesPerBlock_ * ELEMWIDTH;
    T* buffer = static_cast<T*>(copy_home(offset_data));
    int total_elements = this_width / (ELEMWIDTH * sizeof(T));
    int this_blocks = std::min(nBlocks_, blocks);
    av_vSoA.resize(this_blocks);
    // The number of elements and nSplinesPerBock * nBlocks may not be equal
    // Within each block the CUDA data is also in SoA order
    for (int i = 0; i < this_blocks; ++i)
    {
      int elements = std::min(nSplinesPerBlock_, total_elements - i * nSplinesPerBlock_);
      VectorSoAContainer<T, ELEMWIDTH>& vSoA = av_vSoA[i];
      vSoA.resize(elements);
      assert((i * nSplinesPerBlock_ + elements)*ELEMWIDTH*sizeof(T) <= width);
      std::memcpy(vSoA.data(), (void*)(buffer + i * nSplinesPerBlock_) , elements * ELEMWIDTH * sizeof(T));
    }
  }


private:
  T* data;
  int nBlocks_;
  int nSplinesPerBlock_;
  size_t pitch;
  size_t width;
  size_t height;
  size_t max_width_;
  bool owner;
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

  width             = sizeof(T) * nBlocks * ELEMWIDTH * nSplinesPerBlock;
  height            = 1;
  if (max_width_ < width)
  {
    if (data != nullptr)
      cudaFree(data);
    cudaError_t cu_err = cudaMalloc((void**)&data, width);
    if (cu_err != cudaError::cudaSuccess)
      throw std::runtime_error("Failed GPU allocation");
    max_width_ = width;
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
