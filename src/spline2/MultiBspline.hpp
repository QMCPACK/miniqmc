////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by:
// Jeongnim Kim, jeongnim.kim@intel.com, Intel Corp.
// Amrita Mathuriya, amrita.mathuriya@intel.com, Intel Corp.
// Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//
// File created by:
// Jeongnim Kim, jeongnim.kim@intel.com, Intel Corp.
////////////////////////////////////////////////////////////////////////////////
// -*- C++ -*-
/**@file MultiBspline.hpp
 *
 * Master header file to define MultiBspline
 */
#ifndef QMCPLUSPLUS_MULTIEINSPLINE_COMMON_HPP
#define QMCPLUSPLUS_MULTIEINSPLINE_COMMON_HPP

#include <iostream>
#include <spline2/MultiBsplineData.hpp>
#include <stdlib.h>

namespace qmcplusplus
{

template <typename T> struct MultiBspline
{

  /// define the einsplie object type
  using spliner_type = typename bspline_traits<T, 3>::SplineType;
  /// define the real type
  using real_type = typename bspline_traits<T, 3>::real_type;
  /// actual einspline multi-bspline object
  spliner_type *spline_m;

  MultiBspline() : spline_m(nullptr) {}
  MultiBspline(const MultiBspline &in) = delete;
  MultiBspline &operator=(const MultiBspline &in) = delete;

  int num_splines() const
  {
    return (spline_m == nullptr) ? 0 : spline_m->num_splines;
  }

  size_t sizeInByte() const
  {
    return (spline_m == nullptr) ? 0 : spline_m->coefs_size * sizeof(T);
  }

  template <typename PT, typename VT> void evaluate(const PT &r, VT &psi)
  {
    evaluate_v_impl(r[0], r[1], r[2], psi.data(), 0, psi.size());
  }

  template <typename PT, typename VT, typename GT, typename LT>
  inline void evaluate_vgl(const PT &r, VT &psi, GT &grad, LT &lap)
  {
    evaluate_vgl_impl(r[0], r[1], r[2], psi.data(), grad.data(), lap.data(), 0,
                      psi.size());
  }

  template <typename PT, typename VT, typename GT, typename HT>
  inline void evaluate_vgh(const PT &r, VT &psi, GT &grad, HT &hess)
  {
    evaluate_vgh_impl(r[0], r[1], r[2], psi.data(), grad.data(), hess.data(), 0,
                      psi.size());
  }

  /** compute values vals[first,last)
   *
   * The base address for vals, grads and lapl are set by the callers, e.g.,
   * evaluate_vgh(r,psi,grad,hess,ip).
   */
  void evaluate_v_impl(T x, T y, T z, T *restrict vals, int first,
                       int last) const;

  void evaluate_vgl_impl(T x, T y, T z, T *restrict vals, T *restrict grads,
                         T *restrict lapl, int first, int last,
                         size_t out_offset = 0) const;

  void evaluate_vgh_impl(T x, T y, T z, T *restrict vals, T *restrict grads,
                         T *restrict hess, int first, int last,
                         size_t out_offset = 0) const;
};

} /** qmcplusplus namespace */

/// include evaluate_v_impl
#include <spline2/MultiBsplineValue.hpp>

/** choose vgl/vgh, default MultiBsplineStd.hpp based on Ye's BGQ version
 * Only used by tests
 */
#include <spline2/MultiBsplineStd.hpp>

#endif
