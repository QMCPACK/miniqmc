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
  using QMCT = QMCTraits;
  /// define the einspline object type
  using spliner_type = typename bspline_traits<D, T, 3>::SplineType;

  MultiBsplineFuncs() {}
  /// Don't want to deal with this now
  MultiBsplineFuncs(const MultiBsplineFuncs<D,T>& in) = default;
  MultiBsplineFuncs& operator=(const MultiBsplineFuncs<D,T>& in) = delete;

  struct PosBuffer
  {
    PosBuffer() : buffer(nullptr), dev_buffer(nullptr) {}
    ~PosBuffer()
    {
      if (buffer != nullptr)
	delete[] buffer;
      if (dev_buffer != nullptr)
	cudaFree(dev_buffer);
    }
    T* buffer;
    T* dev_buffer;
    T* make(const std::vector<std::array<T,3>>& pos)
    {
      size_t size = 3 * pos.size();
      buffer = new T[size];
      for (int i = 0; i < pos.size(); i++)
	{
	  buffer[i*3+0] = pos[i][0];
	  buffer[i*3+1] = pos[i][1];
	  buffer[i*3+2] = pos[i][2];
	}
      cudaMalloc((void**)&dev_buffer,size*sizeof(T));
      cudaError_t err = cudaMemcpy(dev_buffer,buffer,size*sizeof(T),cudaMemcpyHostToDevice);
      if ( err != cudaSuccess )
	{
	      fprintf (stderr, "Copy of positions to GPU failed.  Error:  %s\n",
	     cudaGetErrorString(err));
    abort();
	}
      return dev_buffer;
    }
  };
  
  void evaluate_v(const spliner_type* spline_m,
			 const std::vector<std::array<T,3>>&,
			 T*&& vals, size_t num_splines);

  void evaluate_vgl(const spliner_type* restrict spline_m,
                    const std::vector<std::array<T,3>>&,
		    T*&& linv,
                    T*&& vals,
                    T*&& lapl,
                    size_t num_splines) const;

  void evaluate_vgh(const typename bspline_traits<D, T, 3>::SplineType* restrict spline_m,
                    const std::vector<std::array<T,3>>&,
                    T*&& vals,
                    T*&& grads,
                    T*&& hess,
                    size_t num_splines) const;
};

template<>
inline void MultiBsplineFuncs<Devices::CUDA, double>::evaluate_v(const multi_UBspline_3d_d<Devices::CUDA>* spline_m, const std::vector<std::array<double,3>>& pos, double*&& vals, size_t num_splines)
{
  PosBuffer pos_d;
  eval_multi_multi_UBspline_3d_d_cuda(spline_m, pos_d.make(pos), vals, num_splines); 
}

template<>
inline void MultiBsplineFuncs<Devices::CUDA,float>::evaluate_v(const typename bspline_traits<Devices::CUDA, float, 3>::SplineType* restrict spline_m, const std::vector<std::array<float,3>>& pos, float*&& vals, size_t num_splines)
{
  PosBuffer pos_f;
  eval_multi_multi_UBspline_3d_s_cuda(spline_m,pos_f.make(pos), vals, num_splines); 
}

template<>
inline void MultiBsplineFuncs<Devices::CUDA, double>::evaluate_vgl(const  MultiBsplineFuncs<Devices::CUDA, double>::spliner_type* restrict spline_m,
                    const std::vector<std::array<double,3>>& pos,
		    double*&& linv,
                    double*&& vals,
                    double*&& lapl,
                    size_t num_splines) const
{
  PosBuffer pos_d;
  int row_stride = 2;
  eval_multi_multi_UBspline_3d_d_vgl_cuda(spline_m, pos_d.make(pos), linv, vals, lapl, num_splines, row_stride);
}

template<>
inline void MultiBsplineFuncs<Devices::CUDA, float>::evaluate_vgl(const  MultiBsplineFuncs<Devices::CUDA, float>::spliner_type* restrict spline_m,
                    const std::vector<std::array<float,3>>& pos,
		    float*&& linv,
                    float*&& vals,
                    float*&& lapl,
                    size_t num_splines) const
{
  PosBuffer pos_f;
  int row_stride = 2;
  eval_multi_multi_UBspline_3d_s_vgl_cuda(spline_m, pos_f.make(pos), linv, &vals, &lapl, num_splines, row_stride);
}

template<>
inline void MultiBsplineFuncs<Devices::CUDA, double>::evaluate_vgh(
    const MultiBsplineFuncs<Devices::CUDA, double>::spliner_type* restrict spline_m,
    const std::vector<std::array<double,3>>& pos,
    double*&& vals,
    double*&& grads,
    double*&& hess,
    size_t num_splines) const
{
}

template<>
inline void MultiBsplineFuncs<Devices::CUDA, float>::evaluate_vgh(
    const MultiBsplineFuncs<Devices::CUDA, float>::spliner_type* restrict spline_m,
    const std::vector<std::array<float,3>>& pos,
    float*&& vals,
    float*&& grads,
    float*&& hess,
    size_t num_splines) const
{
}

  // template<typename T>
// inline void MultiBsplineFuncs<Devices::CUDA, T>::evaluate_v(
//     const MultiBsplineFuncs<Devices::CUDA, T>::spliner_type* restrict spline_m,
//     double x,
//     double y,
//     double z,
//     QMCT::ValueType* restrict vals,
//     size_t num_splines) const
// {
// }

// explicit instantiations
extern template class MultiBsplineFuncs<Devices::CUDA, float>;
extern template class MultiBsplineFuncs<Devices::CUDA, double>;

} // namespace qmcplusplus
#endif
