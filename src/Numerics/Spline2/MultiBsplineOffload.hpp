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
/**@file MultiBsplineOffload.hpp
 *
 * Master header file to define MultiBsplineOffload
 */
#ifndef QMCPLUSPLUS_MULTIEINSPLINE_OFFLOAD_HPP
#define QMCPLUSPLUS_MULTIEINSPLINE_OFFLOAD_HPP
#include "config.h"
#include <iostream>
#include <Numerics/Spline2/MultiBsplineData.hpp>
#include <OpenMP/OMPstd.hpp>
#include <stdlib.h>

namespace qmcplusplus
{

namespace spline2offload
{

template <typename T>
inline void evaluate_v(const typename bspline_traits<T, 3>::SplineType* restrict spline_m,
                                           int ix, int iy, int iz,
                                           const T a[4], const T b[4], const T c[4],
                                           T *restrict vals,
                                           size_t num_splines)
{
  const intptr_t xs = spline_m->x_stride;
  const intptr_t ys = spline_m->y_stride;
  const intptr_t zs = spline_m->z_stride;

  constexpr T zero(0);
  OMPstd::fill_n(vals, num_splines, zero);

  for (size_t i = 0; i < 4; i++)
    for (size_t j = 0; j < 4; j++)
    {
      const T pre00 = a[i] * b[j];
      const T *restrict coefs =
          spline_m->coefs + (ix + i) * xs + (iy + j) * ys + iz * zs;
      for (size_t n = 0; n < num_splines; n++)
        vals[n] +=
            pre00 * (c[0] * coefs[n] + c[1] * coefs[n + zs] +
                     c[2] * coefs[n + 2 * zs] + c[3] * coefs[n + 3 * zs]);
    }
}

template <typename T>
inline void evaluate_v_v2(const typename bspline_traits<T, 3>::SplineType* restrict spline_m,
                                           int ix, int iy, int iz,
                                           const T a[4], const T b[4], const T c[4],
                                           T *restrict vals,
                                           int first, int last)
{
  const intptr_t xs = spline_m->x_stride;
  const intptr_t ys = spline_m->y_stride;
  const intptr_t zs = spline_m->z_stride;

#ifdef ENABLE_OFFLOAD
  #pragma omp for nowait
#else
  #pragma omp simd aligned(vals)
#endif
  for (size_t n = 0; n < last - first; n++)
  {
    T val = T();
    for (size_t i = 0; i < 4; i++)
      for (size_t j = 0; j < 4; j++)
        {
          const T *restrict coefs =
            spline_m->coefs + (ix + i) * xs + (iy + j) * ys + iz * zs + first;
          val += a[i] * b[j] * 
            (c[0] * coefs[n] + c[1] * coefs[n + zs] +
             c[2] * coefs[n + 2 * zs] + c[3] * coefs[n + 3 * zs]);
        }
    vals[n] = val;
  }
}

template <typename T>
inline void
evaluate_vgl(const typename bspline_traits<T, 3>::SplineType* restrict spline_m,
                                 int ix, int iy, int iz,
                                 const T a[4], const T b[4], const T c[4],
                                 const T da[4], const T db[4], const T dc[4],
                                 const T d2a[4], const T d2b[4], const T d2c[4],
                                 T *restrict vals,
                                 T *restrict grads, T *restrict lapl,
                                 size_t num_splines)
{
  const intptr_t xs = spline_m->x_stride;
  const intptr_t ys = spline_m->y_stride;
  const intptr_t zs = spline_m->z_stride;

  const size_t out_offset = spline_m->num_splines;

  T *restrict gx = grads;
  T *restrict gy = grads + out_offset;
  T *restrict gz = grads + 2 * out_offset;
  T *restrict lx = lapl;
  T *restrict ly = lapl + out_offset;
  T *restrict lz = lapl + 2 * out_offset;

  OMPstd::fill_n(vals,  out_offset  , T());
  OMPstd::fill_n(grads, out_offset*3, T());
  OMPstd::fill_n(lapl,  out_offset*3, T());

  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
    {

      const T pre20 = d2a[i] * b[j];
      const T pre10 = da[i] * b[j];
      const T pre00 = a[i] * b[j];
      const T pre11 = da[i] * db[j];
      const T pre01 = a[i] * db[j];
      const T pre02 = a[i] * d2b[j];

      const T *restrict coefs =
          spline_m->coefs + (ix + i) * xs + (iy + j) * ys + iz * zs;
      const T *restrict coefszs  = coefs + zs;
      const T *restrict coefs2zs = coefs + 2 * zs;
      const T *restrict coefs3zs = coefs + 3 * zs;

      for (int n = 0; n < num_splines; n++)
      {
        const T coefsv    = coefs[n];
        const T coefsvzs  = coefszs[n];
        const T coefsv2zs = coefs2zs[n];
        const T coefsv3zs = coefs3zs[n];

        T sum0 = c[0] * coefsv + c[1] * coefsvzs + c[2] * coefsv2zs +
                 c[3] * coefsv3zs;
        T sum1 = dc[0] * coefsv + dc[1] * coefsvzs + dc[2] * coefsv2zs +
                 dc[3] * coefsv3zs;
        T sum2 = d2c[0] * coefsv + d2c[1] * coefsvzs + d2c[2] * coefsv2zs +
                 d2c[3] * coefsv3zs;
        gx[n] += pre10 * sum0;
        gy[n] += pre01 * sum0;
        gz[n] += pre00 * sum1;
        lx[n] += pre20 * sum0;
        ly[n] += pre02 * sum0;
        lz[n] += pre00 * sum2;
        vals[n] += pre00 * sum0;
      }
    }

  const T dxInv = spline_m->x_grid.delta_inv;
  const T dyInv = spline_m->y_grid.delta_inv;
  const T dzInv = spline_m->z_grid.delta_inv;

  const T dxInv2 = dxInv * dxInv;
  const T dyInv2 = dyInv * dyInv;
  const T dzInv2 = dzInv * dzInv;

  for (int n = 0; n < num_splines; n++)
  {
    gx[n] *= dxInv;
    gy[n] *= dyInv;
    gz[n] *= dzInv;
    lx[n] = lx[n] * dxInv2 + ly[n] * dyInv2 + lz[n] * dzInv2;
  }
}

template <typename T>
inline void
evaluate_vgh(const typename bspline_traits<T, 3>::SplineType* restrict spline_m,
                                 int ix, int iy, int iz,
                                 const T a[4], const T b[4], const T c[4],
                                 const T da[4], const T db[4], const T dc[4],
                                 const T d2a[4], const T d2b[4], const T d2c[4],
                                 T *restrict vals,
                                 T *restrict grads, T *restrict hess,
                                 size_t num_splines)
{
  const intptr_t xs = spline_m->x_stride;
  const intptr_t ys = spline_m->y_stride;
  const intptr_t zs = spline_m->z_stride;

  const size_t out_offset = spline_m->num_splines;

  T *restrict gx = grads;
  T *restrict gy = grads + out_offset;
  T *restrict gz = grads + 2 * out_offset;

  T *restrict hxx = hess;
  T *restrict hxy = hess + out_offset;
  T *restrict hxz = hess + 2 * out_offset;
  T *restrict hyy = hess + 3 * out_offset;
  T *restrict hyz = hess + 4 * out_offset;
  T *restrict hzz = hess + 5 * out_offset;

  OMPstd::fill_n(vals,  out_offset  , T());
  OMPstd::fill_n(grads, out_offset*3, T());
  OMPstd::fill_n(hess,  out_offset*6, T());

  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
    {
      const T *restrict coefs =
          spline_m->coefs + (ix + i) * xs + (iy + j) * ys + iz * zs;
      const T *restrict coefszs  = coefs + zs;
      const T *restrict coefs2zs = coefs + 2 * zs;
      const T *restrict coefs3zs = coefs + 3 * zs;

      const T pre20 = d2a[i] * b[j];
      const T pre10 = da[i] * b[j];
      const T pre00 = a[i] * b[j];
      const T pre11 = da[i] * db[j];
      const T pre01 = a[i] * db[j];
      const T pre02 = a[i] * d2b[j];

#ifdef ENABLE_OFFLOAD
      #pragma omp for nowait
#else
      #pragma omp simd aligned(coefs,coefszs,coefs2zs,coefs3zs,vals,gx,gy,gz,hxx,hyy,hzz,hxy,hxz,hyz)
#endif
      for (int n = 0; n < num_splines; n++)
      {

        T coefsv    = coefs[n];
        T coefsvzs  = coefszs[n];
        T coefsv2zs = coefs2zs[n];
        T coefsv3zs = coefs3zs[n];

        T sum0 = c[0] * coefsv + c[1] * coefsvzs + c[2] * coefsv2zs +
                 c[3] * coefsv3zs;
        T sum1 = dc[0] * coefsv + dc[1] * coefsvzs + dc[2] * coefsv2zs +
                 dc[3] * coefsv3zs;
        T sum2 = d2c[0] * coefsv + d2c[1] * coefsvzs + d2c[2] * coefsv2zs +
                 d2c[3] * coefsv3zs;

        hxx[n] += pre20 * sum0;
        hxy[n] += pre11 * sum0;
        hxz[n] += pre10 * sum1;
        hyy[n] += pre02 * sum0;
        hyz[n] += pre01 * sum1;
        hzz[n] += pre00 * sum2;
        gx[n] += pre10 * sum0;
        gy[n] += pre01 * sum0;
        gz[n] += pre00 * sum1;
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

#ifdef ENABLE_OFFLOAD
  #pragma omp for nowait
#else
  #pragma omp simd aligned(gx,gy,gz,hxx,hyy,hzz,hxy,hxz,hyz)
#endif
  for (int n = 0; n < num_splines; n++)
  {
    gx[n] *= dxInv;
    gy[n] *= dyInv;
    gz[n] *= dzInv;
    hxx[n] *= dxx;
    hyy[n] *= dyy;
    hzz[n] *= dzz;
    hxy[n] *= dxy;
    hxz[n] *= dxz;
    hyz[n] *= dyz;
  }
}

template <typename T>
inline void
evaluate_vgh_v2(const typename bspline_traits<T, 3>::SplineType* restrict spline_m,
                                 int ix, int iy, int iz,
                                 const T a[4], const T b[4], const T c[4],
                                 const T da[4], const T db[4], const T dc[4],
                                 const T d2a[4], const T d2b[4], const T d2c[4],
                                 T *restrict val_grad_hess,
                                 size_t out_offset,
                                 const int first,
                                 const int last)
{
  const intptr_t xs = spline_m->x_stride;
  const intptr_t ys = spline_m->y_stride;
  const intptr_t zs = spline_m->z_stride;

#ifdef ENABLE_OFFLOAD
  #pragma omp for nowait
#else
  #pragma omp simd aligned(val_grad_hess)
#endif
  for (int n = 0; n < last - first; n++)
  {
    T val = T();
    T  gx = T();
    T  gy = T();
    T  gz = T();
    T hxx = T();
    T hxy = T();
    T hxz = T();
    T hyy = T();
    T hyz = T();
    T hzz = T();

    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 4; j++)
      {
        const T *restrict coefs =
            spline_m->coefs + (ix + i) * xs + (iy + j) * ys + iz * zs + first;
        const T *restrict coefszs  = coefs + zs;
        const T *restrict coefs2zs = coefs + 2 * zs;
        const T *restrict coefs3zs = coefs + 3 * zs;

        const T pre20 = d2a[i] * b[j];
        const T pre10 = da[i] * b[j];
        const T pre00 = a[i] * b[j];
        const T pre11 = da[i] * db[j];
        const T pre01 = a[i] * db[j];
        const T pre02 = a[i] * d2b[j];

        T coefsv    = coefs[n];
        T coefsvzs  = coefszs[n];
        T coefsv2zs = coefs2zs[n];
        T coefsv3zs = coefs3zs[n];

        T sum0 = c[0] * coefsv + c[1] * coefsvzs + c[2] * coefsv2zs +
                 c[3] * coefsv3zs;
        T sum1 = dc[0] * coefsv + dc[1] * coefsvzs + dc[2] * coefsv2zs +
                 dc[3] * coefsv3zs;
        T sum2 = d2c[0] * coefsv + d2c[1] * coefsvzs + d2c[2] * coefsv2zs +
                 d2c[3] * coefsv3zs;

        hxx += pre20 * sum0;
        hxy += pre11 * sum0;
        hxz += pre10 * sum1;
        hyy += pre02 * sum0;
        hyz += pre01 * sum1;
        hzz += pre00 * sum2;
        gx  += pre10 * sum0;
        gy  += pre01 * sum0;
        gz  += pre00 * sum1;
        val += pre00 * sum0;
      }

    val_grad_hess[n] = val;

    const T dxInv = spline_m->x_grid.delta_inv;
    const T dyInv = spline_m->y_grid.delta_inv;
    const T dzInv = spline_m->z_grid.delta_inv;

    val_grad_hess[n + 1 * out_offset] = gx * dxInv;
    val_grad_hess[n + 2 * out_offset] = gy * dyInv;
    val_grad_hess[n + 3 * out_offset] = gz * dzInv;
    val_grad_hess[n + 4 * out_offset] = hxx * dxInv * dxInv;
    val_grad_hess[n + 5 * out_offset] = hxy * dxInv * dyInv;
    val_grad_hess[n + 6 * out_offset] = hxz * dxInv * dzInv;
    val_grad_hess[n + 7 * out_offset] = hyy * dyInv * dyInv;
    val_grad_hess[n + 8 * out_offset] = hyz * dyInv * dzInv;
    val_grad_hess[n + 9 * out_offset] = hzz * dzInv * dzInv;
  }
}

}

} /** qmcplusplus namespace */
#endif
