//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
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
#ifndef QMCPLUSPLUS_MULTIEINSPLINE_VALUE_HPP
#define QMCPLUSPLUS_MULTIEINSPLINE_VALUE_HPP

namespace qmcplusplus
{

  /** define evaluate: common to any implementation */
  template<typename T>
    inline void
    MultiBspline<T>::evaluate_v_impl(T x, T y, T z, T* restrict vals, int first, int last) const
    {
      x -= spline_m->x_grid.start;
      y -= spline_m->y_grid.start;
      z -= spline_m->z_grid.start;
      T tx,ty,tz;
      int ix,iy,iz;
      SplineBound<T>::get(x*spline_m->x_grid.delta_inv,tx,ix,spline_m->x_grid.num-1);
      SplineBound<T>::get(y*spline_m->y_grid.delta_inv,ty,iy,spline_m->y_grid.num-1);
      SplineBound<T>::get(z*spline_m->z_grid.delta_inv,tz,iz,spline_m->z_grid.num-1);
      T a[4], b[4], c[4];

      MultiBsplineData<T>::compute_prefactors(a, tx);
      MultiBsplineData<T>::compute_prefactors(b, ty);
      MultiBsplineData<T>::compute_prefactors(c, tz);

      const intptr_t xs = spline_m->x_stride;
      const intptr_t ys = spline_m->y_stride;
      const intptr_t zs = spline_m->z_stride;

      CONSTEXPR T zero(0);
      const size_t num_splines=last-first;
      ASSUME_ALIGNED(vals);
      std::fill(vals,vals+num_splines,zero);

      for (size_t i=0; i<4; i++)
        for (size_t j=0; j<4; j++){
          const T pre00 =  a[i]*b[j];
          const T* restrict coefs = spline_m->coefs + ((ix+i)*xs + (iy+j)*ys + iz*zs)+first; ASSUME_ALIGNED(coefs);
//#pragma omp simd
          for(size_t n=0; n<num_splines; n++)
            vals[n] += pre00*(c[0]*coefs[n] + c[1]*coefs[n+zs] + c[2]*coefs[n+2*zs] + c[3]*coefs[n+3*zs]);
        }

    }

}
#endif
