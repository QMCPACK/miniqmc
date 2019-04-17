////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2018 QMCPACK developers.
//
// File developed by:
// Peter Doak, doakpw@ornl.gov, Oak Ridge National Lab
//
// File created by:
// Peter Doak, doakpw@ornl.gov, Oak Ridge National Lab
////////////////////////////////////////////////////////////////////////////////

#ifndef MULTIBSPLINE_CUDA_HPP
#define MULTIBSPLINE_CUDA_HPP

#include <cuda.h>
#include <cuda_runtime_api.h>
#include "Utilities/Configuration.h"
#include "Numerics/Spline2/MultiBsplineData.hpp"
#include "Numerics/Spline2/MultiBsplineFuncs.hpp"
#include "Numerics/Einspline/multi_bspline_eval_cuda.h"

#ifdef __CUDA_ARCH__
#undef ASSUME_ALIGNED
#define ASSUME_ALIGNED(x)
#endif

namespace qmcplusplus
{
/** This specialization takes device pointers
 *  It maintains no state it just holds the evaluates
 */
template<typename T>
struct MultiBsplineFuncs<Devices::CUDA, T>
{
  static constexpr Devices D = Devices::CUDA;
  using QMCT                 = QMCTraits;
  /// define the einspline object type
  using spliner_type = typename bspline_traits<D, T, 3>::SplineType;

  MultiBsplineFuncs(int pos_block_size = 1) : pos_block_size_(pos_block_size)
  {
    pos_buf_.resize(pos_block_size);
  }
  /// Don't want to deal with this now
  MultiBsplineFuncs(const MultiBsplineFuncs<D, T>& in) = default;
  MultiBsplineFuncs& operator=(const MultiBsplineFuncs<D, T>& in) = delete;

    /** This could be a general small cuda parameter buffer except for the 3
     */
  template<typename T_>
  struct PosBuffer
  {
     PosBuffer() : stream_(cudaStreamPerThread), buffer(nullptr), dev_buffer(nullptr), size_(0) {}
      PosBuffer(cudaStream_t stream) : stream_(stream), buffer(nullptr), dev_buffer(nullptr), size_(0) {}
     
    ~PosBuffer()
    {
      if (buffer != nullptr) cudaFreeHost(buffer);
      if (dev_buffer != nullptr) cudaFree(dev_buffer);
    }

    void operator() (cudaStream_t stream) { stream_ = stream; }
    void resize(int size)
    {
      if (size > size_)
      {
	  if (buffer != nullptr) cudaFreeHost(buffer);
	  if (dev_buffer != nullptr) cudaFree(dev_buffer);
	  cudaMallocHost((void**)&buffer, size );
	  cudaMalloc((void**)&dev_buffer, size);
	  size_ = size;
      }
    }
	      
    T_* make(const std::vector<std::array<T_, 3>>& pos)
    {
      int n_pos = pos.size();
      size_t byte_size = 3 * n_pos * sizeof(T_);
      resize(byte_size);
      
      for (int i = 0; i < n_pos; i++)
      {
        buffer[i * 3 + 0] = pos[i][0];
        buffer[i * 3 + 1] = pos[i][1];
        buffer[i * 3 + 2] = pos[i][2];
      }
      
      cudaError_t err = cudaMemcpyAsync(dev_buffer, buffer, byte_size, cudaMemcpyDefault, stream_);
      if (err != cudaSuccess)
      {
        fprintf(stderr, "Copy of positions to GPU failed.  Error:  %s\n", cudaGetErrorString(err));
        abort();
      }
      return dev_buffer;
    }
  private:
    T_* buffer;
    T_* dev_buffer;
    size_t size_;  //in bytes
    cudaStream_t stream_;

  };

  
  // void evaluate_v(const spliner_type* spline_m,
  // 			 const std::vector<std::array<T,3>>&,
  // 		         GPUArray<T,1>& vals, size_t num_splines);

  void evaluate_v(const spliner_type* spline_m, const std::vector<std::array<T, 3>>&, T* vals,
                  int num_blocks, size_t num_splines, size_t spline_block_size);

  void evaluate_vgl(const spliner_type* restrict spline_m,
                    const std::vector<std::array<T, 3>>&,
                    T* linv,
                    T* vals,
                    T* lapl,
                    size_t num_splines);

  void evaluate_vgh(const typename bspline_traits<D, T, 3>::SplineType* restrict spline_m,
                    const std::vector<std::array<T, 3>>&,
                    T* vals,
                    T* grads,
                    T* hess,
		    int num_blocks,
		    size_t num_splines,
                    size_t spline_block_size,
                    cudaStream_t stream = cudaStreamPerThread);

private:
  PosBuffer<T> pos_buf_;
  int pos_block_size_;
};

/** New implementation with non const spline block sizeo
 */
template<>
inline void
MultiBsplineFuncs<Devices::CUDA, double>::evaluate_v(const multi_UBspline_3d_d<Devices::CUDA>* spline_m,
                                                     const std::vector<std::array<double, 3>>& pos,
                                                     double* vals, int num_blocks, size_t num_splines,
                                                     size_t spline_block_size)
{
  eval_multi_multi_UBspline_3d_d_cuda(spline_m, pos_buf_.make(pos), vals, num_blocks, spline_block_size, pos.size());
}

template<>
inline void MultiBsplineFuncs<Devices::CUDA, float>::evaluate_v(
    const typename bspline_traits<Devices::CUDA, float, 3>::SplineType* restrict spline_m,
    const std::vector<std::array<float, 3>>& pos, float* vals, int num_blocks, size_t num_splines,
    size_t spline_block_size)
{
  eval_multi_multi_UBspline_3d_s_cuda(spline_m, pos_buf_.make(pos), vals, num_splines);
}

/** New implementation with non const spline block size
 *  with nblocks and splineblock size this whole thing is over specified
 */
template<>
inline void MultiBsplineFuncs<Devices::CUDA, double>::evaluate_vgh(
    const MultiBsplineFuncs<Devices::CUDA, double>::spliner_type* restrict spline_m,
    const std::vector<std::array<double, 3>>& pos,
    double* vals,
    double* grads,
    double* hess,
    int num_blocks, //blocks per participant
    size_t num_splines,
    size_t spline_block_size,
    cudaStream_t stream)
{
  pos_buf_(stream);
  // This is a bit of legacy, the implementation should be aware of this and pass it
  if (spline_block_size == 0) spline_block_size = spline_m->num_splines;

  eval_multi_multi_UBspline_3d_d_vgh_cuda(spline_m, pos_buf_.make(pos), vals, grads, hess, num_blocks,
                                          spline_block_size, pos.size(), stream);
}

// template<>
// inline void MultiBsplineFuncs<Devices::CUDA, float>::evaluate_vgh(std::vector<SPOSet*> spos,
// 								  std::vector<std::array<float, 3>>& pos)
// {
//   //for now there is no support for group eval of walkers with different einspline basis.
  
// }

template<>
inline void MultiBsplineFuncs<Devices::CUDA, float>::evaluate_vgh(
    const MultiBsplineFuncs<Devices::CUDA, float>::spliner_type* restrict spline_m,
    const std::vector<std::array<float, 3>>& pos,
    float* vals,
    float* grads,
    float* hess,
    int num_blocks,
    size_t num_splines,
    size_t spline_block_size,
    cudaStream_t stream)
{
  pos_buf_(stream);

  if (spline_block_size == 0) spline_block_size = spline_m->num_splines;
  eval_multi_multi_UBspline_3d_s_vgh_cuda(spline_m, pos_buf_.make(pos), vals, grads, hess, num_splines);
}

template<>
inline void MultiBsplineFuncs<Devices::CUDA, double>::evaluate_vgl(
    const MultiBsplineFuncs<Devices::CUDA, double>::spliner_type* restrict spline_m,
    const std::vector<std::array<double, 3>>& pos,
    double* linv,
    double* vals,
    double* lapl,
    size_t num_splines)
{
  int row_stride = 2;
  eval_multi_multi_UBspline_3d_d_vgl_cuda(spline_m, pos_buf_.make(pos), linv, vals, lapl, num_splines,
                                          row_stride);
}

template<>
inline void MultiBsplineFuncs<Devices::CUDA, float>::evaluate_vgl(
    const MultiBsplineFuncs<Devices::CUDA, float>::spliner_type* restrict spline_m,
    const std::vector<std::array<float, 3>>& pos,
    float* linv,
    float* vals,
    float* lapl,
    size_t num_splines)
{
  int row_stride = 2;
  eval_multi_multi_UBspline_3d_s_vgl_cuda(spline_m, pos_buf_.make(pos), linv, vals, lapl, num_splines,
                                          row_stride);
}


// explicit instantiations
extern template class MultiBsplineFuncs<Devices::CUDA, float>;
extern template class MultiBsplineFuncs<Devices::CUDA, double>;

} // namespace qmcplusplus
#endif
