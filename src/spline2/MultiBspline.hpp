//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by: Jeongnim Kim, jeongnim.kim@intel.com, Intel Corp.
//                    Amrita Mathuriya, amrita.mathuriya@intel.com, Intel Corp.
//                    Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//
// File created by: Jeongnim Kim, jeongnim.kim@intel.com, Intel Corp.
//////////////////////////////////////////////////////////////////////////////////////
// -*- C++ -*-
/**@file MultiBspline.hpp
 *
 * Master header file to define MultiBspline
 */
#ifndef QMCPLUSPLUS_MULTIEINSPLINE_COMMON_HPP
#define QMCPLUSPLUS_MULTIEINSPLINE_COMMON_HPP

#include <iostream>
#include <spline2/bspline_allocator.hpp>
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
  /// set to true if create is invoked
  bool own_spline;
  /// actual einspline multi-bspline object
  spliner_type *spline_m;
  /// offset
  std::vector<int> offset;
  /// use allocator
  einspline::Allocator myAllocator;

  MultiBspline() : own_spline(false), spline_m(nullptr) {}
  MultiBspline(const MultiBspline &in) = delete;
  MultiBspline &operator=(const MultiBspline &in) = delete;

  ~MultiBspline()
  {
    if (own_spline)
    {
      myAllocator.destroy(spline_m);
    }
  }

  template <typename RV, typename IV>
  void create(RV &start, RV &end, IV &ng, bc_code bc, int num_splines,
              int nteams = 1)
  {
    if (spline_m == nullptr)
    {
      spline_m =
          myAllocator.createMultiBspline(T(0), start, end, ng, bc, num_splines);
      own_spline = true;
    }
    // should be refined to ensure alignment with minimal waste
    int nsb = num_splines / nteams;
    offset.resize(nteams + 1);
    for (int i = 0; i < nteams; ++i) offset[i] = i * nsb;
    offset[nteams]                             = num_splines;
  }

  /** create the einspline as used in the builder
   */
  template <typename GT, typename BCT>
  void create(GT &grid, BCT &bc, int num_splines, int nteams = 1)
  {
    if (spline_m == nullptr)
    {
      typename bspline_traits<T, 3>::BCType xBC, yBC, zBC;
      xBC.lCode = bc[0].lCode;
      yBC.lCode = bc[1].lCode;
      zBC.lCode = bc[2].lCode;
      xBC.rCode = bc[0].rCode;
      yBC.rCode = bc[1].rCode;
      zBC.rCode = bc[2].rCode;
      xBC.lVal  = static_cast<T>(bc[0].lVal);
      yBC.lVal  = static_cast<T>(bc[1].lVal);
      zBC.lVal  = static_cast<T>(bc[2].lVal);
      xBC.rVal  = static_cast<T>(bc[0].rVal);
      yBC.rVal  = static_cast<T>(bc[1].rVal);
      zBC.rVal  = static_cast<T>(bc[2].rVal);
      spline_m  = myAllocator.allocateMultiBspline(grid[0], grid[1], grid[2],
                                                  xBC, yBC, zBC, num_splines);
      own_spline = true;
    }
    // should be refined to ensure alignment with minimal waste
    int nsb = num_splines / nteams;
    offset.resize(nteams + 1);
    for (int i = 0; i < nteams; ++i) offset[i] = i * nsb;
    offset[nteams]                             = num_splines;
  }

  void flush_zero() const
  {
    if (spline_m != nullptr)
      std::fill(spline_m->coefs, spline_m->coefs + spline_m->coefs_size, T(0));
  }

  int num_splines() const
  {
    return (spline_m == nullptr) ? 0 : spline_m->num_splines;
  }

  size_t sizeInByte() const
  {
    return (spline_m == nullptr) ? 0 : spline_m->coefs_size * sizeof(T);
  }

  template <typename CT> inline void set(int i, CT &data)
  {
    myAllocator.set(data.data(), spline_m, i);
  }

  /** copy a single spline to the big table
   * @param aSpline UBspline_3d_(d,s)
   * @param int index of aSpline
   * @param offset_ starting index for the case of multiple domains
   * @param base_ number of bases
   */
  template <typename SingleSpline>
  void copy_spline(SingleSpline *aSpline, int i, const int *offset_,
                   const int *base_)
  {
    myAllocator.copy(aSpline, spline_m, i, offset_, base_);
  }

  template <typename PT, typename VT> void evaluate(const PT &r, VT &psi)
  {
    evaluate_v_impl(r[0], r[1], r[2], psi.data(), 0, psi.size());
  }

  template <typename PT, typename VT>
  void evaluate(const PT &r, VT &psi, int ip)
  {
    const int first = offset[ip];
    evaluate_v_impl(r[0], r[1], r[2], psi.data() + first, first,
                    offset[ip + 1]);
  }

  template <typename PT, typename VT, typename GT>
  inline void evaluate(const PT &r, VT &psi, GT &grad)
  {
    // einspline::evaluate(spliner,r,psi,grad);
  }

  template <typename PT, typename VT, typename GT, typename LT>
  inline void evaluate_vgl(const PT &r, VT &psi, GT &grad, LT &lap)
  {
    evaluate_vgl_impl(r[0], r[1], r[2], psi.data(), grad.data(), lap.data(), 0,
                      psi.size());
  }

  template <typename PT, typename VT, typename GT, typename LT>
  inline void evaluate_vgl(const PT &r, VT &psi, GT &grad, LT &lap, int ip)
  {
    const int first = offset[ip];
    evaluate_vgl_impl(r[0], r[1], r[2], psi.data() + first, grad.data() + first,
                      lap.data() + first, first, offset[ip + 1]);
  }

  template <typename PT, typename VT, typename GT, typename HT>
  inline void evaluate_vgh(const PT &r, VT &psi, GT &grad, HT &hess)
  {
    evaluate_vgh_impl(r[0], r[1], r[2], psi.data(), grad.data(), hess.data(), 0,
                      psi.size());
  }

  template <typename PT, typename VT, typename GT, typename HT>
  inline void evaluate_vgh(const PT &r, VT &psi, GT &grad, HT &hess, int ip)
  {
    const int first = offset[ip];
    evaluate_vgh_impl(r[0], r[1], r[2], psi.data() + first, grad.data() + first,
                      hess.data() + first, first, offset[ip + 1]);
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
