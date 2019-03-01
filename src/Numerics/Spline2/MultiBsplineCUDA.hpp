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
  using QMCT                 = QMCTraits;
  /// define the einspline object type
  using spliner_type = typename bspline_traits<D, T, 3>::SplineType;

  MultiBspline() {}
  /// Don't want to deal with this now
  MultiBspline(const MultiBspline<D, T>& in) = default;
  MultiBspline& operator=(const MultiBspline<D, T>& in) = delete;

  void evaluate_v(const spliner_type* restrict spline_m, const std::vector<std::array<T, 3>>& pos,
                  T* vals, size_t num_splines) const;

  void evaluate_vgl(const spliner_type* restrict spline_m,
                    const std::vector<std::array<T, 3>>& pos,
                    T* linv,
                    T* vals,
                    T* lapl,
                    size_t num_splines) const;

  void evaluate_vgh(const typename bspline_traits<D, T, 3>::SplineType* restrict spline_m,
                    const std::Vector<std::array<T, 3>>& pos,
                    T* vals,
                    T* grads,
                    T* hess,
                    size_t num_splines) const;
};

template<>
inline void MultiBspline<Devices::CUDA, double>::evaluate_v(
    const multi_UBspline_3d_d<Devices::CUDA>* restrict spline_m,
    const std::vector<std::array<T, 3>>& pos, double* vals, size_t num) const
{
  double pos_d[] = {x, y, z};
  eval_multi_multi_UBspline_3d_d_cuda(spline_m, pos_d, &vals, num_splines);
}

template<>
inline void MultiBspline<Devices::CUDA, float>::evaluate_v(
    const typename bspline_traits<Devices::CUDA, float, 3>::SplineType* restrict spline_m,
    const std::vector<std::array<T, 3>>& pos, float* vals, size_t num_splines) const
{
  float pos_d[] = {(float)x, (float)y, (float)z};
  eval_multi_multi_UBspline_3d_s_cuda(spline_m, pos_d, &vals, num_splines);
}

template<>
inline void MultiBspline<Devices::CUDA, double>::evaluate_vgl(
    const MultiBspline<Devices::CUDA, double>::spliner_type* restrict spline_m,
    const std::vector<std::array<T, 3>>& pos,
    double* linv,
    double* vals,
    double* lapl,
    size_t num_splines) const
{
  double pos_d[] = {y, y, z};
  int row_stride = 2;
  eval_multi_multi_UBspline_3d_d_vgl_cuda(spline_m, pos_d, linv, &vals, &lapl, num_splines,
                                          row_stride);
}

template<>
inline void MultiBspline<Devices::CUDA, float>::evaluate_vgl(
    const MultiBspline<Devices::CUDA, float>::spliner_type* restrict spline_m,
const std::vector<std::array<T, 3>>& pos,    float* linv,
    float* vals,
    float* lapl,
    size_t num_splines) const
{
  float pos_f[]  = {(float)x, (float)y, (float)z};
  int row_stride = 2;
  eval_multi_multi_UBspline_3d_s_vgl_cuda(spline_m, pos_f, linv, &vals, &lapl, num_splines,
                                          row_stride);
}

template<>
inline void MultiBspline<Devices::CUDA, double>::evaluate_vgh(
    const MultiBspline<Devices::CUDA, double>::spliner_type* restrict spline_m,
const std::vector<std::array<T, 3>>& pos,
    double* vals,
    double* grads,
    double* hess,
    size_t num_splines) const
{}

template<>
inline void MultiBspline<Devices::CUDA, float>::evaluate_vgh(
    const MultiBspline<Devices::CUDA, float>::spliner_type* restrict spline_m,
const std::vector<std::array<T, 3>>& pos,    float* vals,
    float* grads,
    float* hess,
    size_t num_splines) const
{}

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

// explicit instantiations
extern template class MultiBspline<Devices::CUDA, float>;
extern template class MultiBspline<Devices::CUDA, double>;

} // namespace qmcplusplus
#endif
