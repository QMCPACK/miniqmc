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
#include "Numerics/Spline2/MultiBspline.hpp"
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
struct MultiBspline<Devices::CUDA, T>
{
  static constexpr Devices D = Devices::CUDA;
  using QMCT = QMCTraits;
  /// define the einspline object type
  using spliner_type = typename bspline_traits<D, T, 3>::SplineType;

  MultiBspline() {}
  /// Don't want to deal with this now
  MultiBspline(const MultiBspline<D,T>& in) = default;
  MultiBspline& operator=(const MultiBspline<D,T>& in) = delete;

  void evaluate_v(const spliner_type* restrict spline_m,
			 double x,
			 double y,
			 double z,
			 T* vals, size_t num_splines) const;

  void evaluate_vgl(const spliner_type* restrict spline_m,
                    double x,
                    double y,
                    double z,
		    T* linv,
                    T* vals,
                    T* grads,
                    T* lapl,
                    size_t num_splines) const;


  void evaluate_vgh(const spliner_type* restrict spline_m,
                    double x,
                    double y,
                    double z,
                    T* vals,
                    T* grads,
                    T* hess,
                    size_t num_splines) const;
};

// template<typename T>
// inline void MultiBspline<Devices::CUDA, T>::evaluate_v(
//     const MultiBspline<Devices::CUDA, T>::spliner_type* restrict spline_m,
//     double x,
//     double y,
//     double z,
//     QMCT::ValueType* restrict vals,
//     size_t num_splines) const
// {
// }


} // namespace qmcplusplus
#endif
