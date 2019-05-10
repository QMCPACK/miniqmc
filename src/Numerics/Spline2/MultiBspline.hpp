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
 *
 * Contains 3D spline evaluation routines.
 */
#ifndef QMCPLUSPLUS_MULTIEINSPLINE_COMMON_HPP
#define QMCPLUSPLUS_MULTIEINSPLINE_COMMON_HPP

#include <iostream>
#include <Numerics/Spline2/MultiBsplineData.hpp>
#include <Numerics/Spline2/MultiBsplineEvalHelper.hpp>
#include <stdlib.h>

namespace qmcplusplus
{
template<typename T>
struct MultiBspline
{
  /// define the einspline object type
  using spliner_type = typename bspline_traits<T, 3>::SplineType;

  MultiBspline() {}
  MultiBspline(const MultiBspline& in) = delete;
  MultiBspline& operator=(const MultiBspline& in) = delete;

  /** compute values vals[0,num_splines)
   *
   * The base address for vals, grads and lapl are set by the callers, e.g.,
   * evaluate_vgh(r,psi,grad,hess,ip).
   */
  void evaluate_v(const spliner_type* restrict spline_m, T x, T y, T z, T* restrict vals,
                  size_t num_splines) const;

  void evaluate_vgl(const spliner_type* restrict spline_m, T x, T y, T z, T* restrict vals,
                    T* restrict grads, T* restrict lapl, size_t num_splines) const;

  void evaluate_vgh(const spliner_type* restrict spline_m, T x, T y, T z, T* restrict vals,
                    T* restrict grads, T* restrict hess, size_t num_splines) const;
};

template<typename T>
inline void MultiBspline<T>::evaluate_v(const spliner_type* restrict spline_m, T x, T y, T z,
                                        T* restrict vals, size_t num_splines) const
{
  int ix, iy, iz;
  T a[4], b[4], c[4];

  spline2::computeLocationAndFractional(spline_m, x, y, z, ix, iy, iz, a, b, c);

  const intptr_t xs = spline_m->x_stride;
  const intptr_t ys = spline_m->y_stride;
  const intptr_t zs = spline_m->z_stride;

  constexpr T zero(0);
  ASSUME_ALIGNED(vals);
  std::fill(vals, vals + num_splines, zero);

  for (size_t i = 0; i < 4; i++)
    for (size_t j = 0; j < 4; j++)
    {
      const T pre00           = a[i] * b[j];
      const T* restrict coefs = spline_m->coefs + ((ix + i) * xs + (iy + j) * ys + iz * zs);
      ASSUME_ALIGNED(coefs);
      //#pragma omp simd
      for (size_t n = 0; n < num_splines; n++)
        vals[n] += pre00 *
            (c[0] * coefs[n] + c[1] * coefs[n + zs] + c[2] * coefs[n + 2 * zs] +
             c[3] * coefs[n + 3 * zs]);
    }
}

template<typename T>
inline void
MultiBspline<T>::evaluate_vgl(const spliner_type* restrict spline_m, T x, T y, T z, T* restrict vals,
                              T* restrict grads, T* restrict lapl, size_t num_splines) const
{
  int ix, iy, iz;
  T a[4], b[4], c[4], da[4], db[4], dc[4], d2a[4], d2b[4], d2c[4];

  spline2::computeLocationAndFractional(spline_m, x, y, z, ix, iy, iz, a, b, c, da, db, dc, d2a, d2b, d2c);

  const intptr_t xs = spline_m->x_stride;
  const intptr_t ys = spline_m->y_stride;
  const intptr_t zs = spline_m->z_stride;

  const size_t out_offset = spline_m->num_splines;

  ASSUME_ALIGNED(vals);
  T* restrict gx = grads;
  ASSUME_ALIGNED(gx);
  T* restrict gy = grads + out_offset;
  ASSUME_ALIGNED(gy);
  T* restrict gz = grads + 2 * out_offset;
  ASSUME_ALIGNED(gz);
  T* restrict lx = lapl;
  ASSUME_ALIGNED(lx);
  T* restrict ly = lapl + out_offset;
  ASSUME_ALIGNED(ly);
  T* restrict lz = lapl + 2 * out_offset;
  ASSUME_ALIGNED(lz);

  std::fill(vals, vals + num_splines, T());
  std::fill(gx, gx + num_splines, T());
  std::fill(gy, gy + num_splines, T());
  std::fill(gz, gz + num_splines, T());
  std::fill(lx, lx + num_splines, T());
  std::fill(ly, ly + num_splines, T());
  std::fill(lz, lz + num_splines, T());

  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
    {
      const T pre20 = d2a[i] * b[j];
      const T pre10 = da[i] * b[j];
      const T pre00 = a[i] * b[j];
      const T pre11 = da[i] * db[j];
      const T pre01 = a[i] * db[j];
      const T pre02 = a[i] * d2b[j];

      const T* restrict coefs = spline_m->coefs + ((ix + i) * xs + (iy + j) * ys + iz * zs);
      ASSUME_ALIGNED(coefs);
      const T* restrict coefszs = coefs + zs;
      ASSUME_ALIGNED(coefszs);
      const T* restrict coefs2zs = coefs + 2 * zs;
      ASSUME_ALIGNED(coefs2zs);
      const T* restrict coefs3zs = coefs + 3 * zs;
      ASSUME_ALIGNED(coefs3zs);

#pragma noprefetch
#pragma omp simd
      for (int n = 0; n < num_splines; n++)
      {
        const T coefsv    = coefs[n];
        const T coefsvzs  = coefszs[n];
        const T coefsv2zs = coefs2zs[n];
        const T coefsv3zs = coefs3zs[n];

        T sum0 = c[0] * coefsv + c[1] * coefsvzs + c[2] * coefsv2zs + c[3] * coefsv3zs;
        T sum1 = dc[0] * coefsv + dc[1] * coefsvzs + dc[2] * coefsv2zs + dc[3] * coefsv3zs;
        T sum2 = d2c[0] * coefsv + d2c[1] * coefsvzs + d2c[2] * coefsv2zs + d2c[3] * coefsv3zs;
        gx[n]   += pre10 * sum0;
        gy[n]   += pre01 * sum0;
        gz[n]   += pre00 * sum1;
        lx[n]   += pre20 * sum0;
        ly[n]   += pre02 * sum0;
        lz[n]   += pre00 * sum2;
        vals[n] += pre00 * sum0;
      }
    }

  const T dxInv = spline_m->x_grid.delta_inv;
  const T dyInv = spline_m->y_grid.delta_inv;
  const T dzInv = spline_m->z_grid.delta_inv;

  const T dxInv2 = dxInv * dxInv;
  const T dyInv2 = dyInv * dyInv;
  const T dzInv2 = dzInv * dzInv;

#pragma omp simd
  for (int n = 0; n < num_splines; n++)
  {
    gx[n] *= dxInv;
    gy[n] *= dyInv;
    gz[n] *= dzInv;
    lx[n] = lx[n] * dxInv2 + ly[n] * dyInv2 + lz[n] * dzInv2;
  }
}

template<typename T>
inline void
MultiBspline<T>::evaluate_vgh(const spliner_type* restrict spline_m, T x, T y, T z, T* restrict vals,
                              T* restrict grads, T* restrict hess, size_t num_splines) const
{
  int ix, iy, iz;
  T a[4], b[4], c[4], da[4], db[4], dc[4], d2a[4], d2b[4], d2c[4];

  spline2::computeLocationAndFractional(spline_m, x, y, z, ix, iy, iz, a, b, c, da, db, dc, d2a, d2b, d2c);

  const intptr_t xs = spline_m->x_stride;
  const intptr_t ys = spline_m->y_stride;
  const intptr_t zs = spline_m->z_stride;

  const size_t out_offset = spline_m->num_splines;

  ASSUME_ALIGNED(vals);
  T* restrict gx = grads;
  ASSUME_ALIGNED(gx);
  T* restrict gy = grads + out_offset;
  ASSUME_ALIGNED(gy);
  T* restrict gz = grads + 2 * out_offset;
  ASSUME_ALIGNED(gz);

  T* restrict hxx = hess;
  ASSUME_ALIGNED(hxx);
  T* restrict hxy = hess + out_offset;
  ASSUME_ALIGNED(hxy);
  T* restrict hxz = hess + 2 * out_offset;
  ASSUME_ALIGNED(hxz);
  T* restrict hyy = hess + 3 * out_offset;
  ASSUME_ALIGNED(hyy);
  T* restrict hyz = hess + 4 * out_offset;
  ASSUME_ALIGNED(hyz);
  T* restrict hzz = hess + 5 * out_offset;
  ASSUME_ALIGNED(hzz);

  std::fill(vals, vals + num_splines, T());
  std::fill(gx, gx + num_splines, T());
  std::fill(gy, gy + num_splines, T());
  std::fill(gz, gz + num_splines, T());
  std::fill(hxx, hxx + num_splines, T());
  std::fill(hxy, hxy + num_splines, T());
  std::fill(hxz, hxz + num_splines, T());
  std::fill(hyy, hyy + num_splines, T());
  std::fill(hyz, hyz + num_splines, T());
  std::fill(hzz, hzz + num_splines, T());

  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
    {
      const T* restrict coefs = spline_m->coefs + ((ix + i) * xs + (iy + j) * ys + iz * zs);
      ASSUME_ALIGNED(coefs);
      const T* restrict coefszs = coefs + zs;
      ASSUME_ALIGNED(coefszs);
      const T* restrict coefs2zs = coefs + 2 * zs;
      ASSUME_ALIGNED(coefs2zs);
      const T* restrict coefs3zs = coefs + 3 * zs;
      ASSUME_ALIGNED(coefs3zs);

      const T pre20 = d2a[i] * b[j];
      const T pre10 = da[i] * b[j];
      const T pre00 = a[i] * b[j];
      const T pre11 = da[i] * db[j];
      const T pre01 = a[i] * db[j];
      const T pre02 = a[i] * d2b[j];

      const int iSplitPoint = num_splines;
#pragma omp simd
      for (int n = 0; n < iSplitPoint; n++)
      {
        T coefsv    = coefs[n];
        T coefsvzs  = coefszs[n];
        T coefsv2zs = coefs2zs[n];
        T coefsv3zs = coefs3zs[n];

        T sum0 = c[0] * coefsv + c[1] * coefsvzs + c[2] * coefsv2zs + c[3] * coefsv3zs;
        T sum1 = dc[0] * coefsv + dc[1] * coefsvzs + dc[2] * coefsv2zs + dc[3] * coefsv3zs;
        T sum2 = d2c[0] * coefsv + d2c[1] * coefsvzs + d2c[2] * coefsv2zs + d2c[3] * coefsv3zs;

        hxx[n]  += pre20 * sum0;
        hxy[n]  += pre11 * sum0;
        hxz[n]  += pre10 * sum1;
        hyy[n]  += pre02 * sum0;
        hyz[n]  += pre01 * sum1;
        hzz[n]  += pre00 * sum2;
        gx[n]   += pre10 * sum0;
        gy[n]   += pre01 * sum0;
        gz[n]   += pre00 * sum1;
        vals[n] += pre00 * sum0;
      }
    }

  const T dxInv = spline_m->x_grid.delta_inv;
  const T dyInv = spline_m->y_grid.delta_inv;
  const T dzInv = spline_m->z_grid.delta_inv;
  const T dxx   = dxInv * dxInv;
  const T dyy   = dyInv * dyInv;
  const T dzz   = dzInv * dzInv;
  const T dxy   = dxInv * dyInv;
  const T dxz   = dxInv * dzInv;
  const T dyz   = dyInv * dzInv;

#pragma omp simd
  for (int n = 0; n < num_splines; n++)
  {
    gx[n]  *= dxInv;
    gy[n]  *= dyInv;
    gz[n]  *= dzInv;
    hxx[n] *= dxx;
    hyy[n] *= dyy;
    hzz[n] *= dzz;
    hxy[n] *= dxy;
    hxz[n] *= dxz;
    hyz[n] *= dyz;
  }
}

} // namespace qmcplusplus
#endif
