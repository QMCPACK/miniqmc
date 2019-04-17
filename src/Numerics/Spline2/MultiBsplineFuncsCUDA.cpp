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

#include "MultiBsplineFuncsCUDA.hpp"
#include "Numerics/Einspline/multi_bspline_eval_cuda.h"

namespace qmcplusplus
{
// template<>
// inline void MultiBspline<Devices::CUDA, double>::evaluate_vgl(
//     const MultiBspline<Devices::CUDA, double>::spliner_type* restrict spline_m,
//     double x,
//     double y,
//     double z,
//     double* linv,
//     double* vals,
//     double* grads,
//     double* lapl,
//     size_t num_splines) const
// {
//   double pos_d[] = {x,y,z};
// 	     eval_multi_multi_UBspline_3d_d_vgl_cuda(spline_m, pos_d, linv, &vals, &grads, &lapl, num_splines);
// }

// template<T>
// inline void MultiBspline<Devices::CUDA, float>::evaluate_vgl(
//     const MultiBspline<Devices::CUDA, float>::spliner_type* restrict spline_m,
//     double x,
//     double y,
//     double z,
//     float* linv,
//     float* vals,
//     float* grads,
//     float* lapl,
//     size_t num_splines) const
// {
//   float pos_d[] = {(float)x,
// 		   (float)y,
// 		   (float)z};
//   eval_multi_multi_UBspline_3d_s_vgl_cuda(spline_m,pos_d, linv, &vals, &grads, &lapl, num_splines);
// }


template class MultiBsplineFuncs<Devices::CUDA, float>;
template class MultiBsplineFuncs<Devices::CUDA, double>;
} // namespace qmcplusplus
