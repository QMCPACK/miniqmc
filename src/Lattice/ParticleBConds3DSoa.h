////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by:
// Jeongnim Kim, jeongnim.kim@intel.com,
//    Intel Corp.
// Amrita Mathuriya, amrita.mathuriya@intel.com,
//    Intel Corp.
//
// File created by:
// Jeongnim Kim, jeongnim.kim@intel.com,
//    Intel Corp.
////////////////////////////////////////////////////////////////////////////////
// -*- C++ -*-
#ifndef QMCPLUSPLUS_PARTICLE_BCONDS_3D_SOA_H
#define QMCPLUSPLUS_PARTICLE_BCONDS_3D_SOA_H

#include <Lattice/CrystalLattice.h>
#include <OhmmsSoA/VectorSoaContainer.h>
#include <algorithm>
#include <config.h>

namespace qmcplusplus
{

/** specialization for a periodic 3D general cell
 *
 * Wigner-Seitz cell radius > simulation cell radius
 * Need to check image cells
*/
template <class T> struct DTD_BConds<T, 3, PPPG + SOA_OFFSET>
{
  T g00, g10, g20, g01, g11, g21, g02, g12, g22;
  T r00, r10, r20, r01, r11, r21, r02, r12, r22;
  VectorSoaContainer<T, 3> corners;

  DTD_BConds(const CrystalLattice<T, 3> &lat)
  {
    TinyVector<TinyVector<T, 3>, 3> rb;
    rb[0] = lat.a(0);
    rb[1] = lat.a(1);
    rb[2] = lat.a(2);
    find_reduced_basis(rb);

    r00 = rb[0][0];
    r10 = rb[1][0];
    r20 = rb[2][0];
    r01 = rb[0][1];
    r11 = rb[1][1];
    r21 = rb[2][1];
    r02 = rb[0][2];
    r12 = rb[1][2];
    r22 = rb[2][2];

    Tensor<T, 3> rbt;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) rbt(i, j) = rb[i][j];
    Tensor<T, 3> g                          = inverse(rbt);
    g00 = g(0);
    g10 = g(3);
    g20 = g(6);
    g01 = g(1);
    g11 = g(4);
    g21 = g(7);
    g02 = g(2);
    g12 = g(5);
    g22 = g(8);

    CONSTEXPR T minusone(-1);
    CONSTEXPR T zero(0);

    corners.resize(8);
    corners(0) = zero;
    corners(1) = minusone * (rb[0]);
    corners(2) = minusone * (rb[1]);
    corners(3) = minusone * (rb[2]);
    corners(4) = minusone * (rb[0] + rb[1]);
    corners(5) = minusone * (rb[0] + rb[2]);
    corners(6) = minusone * (rb[1] + rb[2]);
    corners(7) = minusone * (rb[0] + rb[1] + rb[2]);
  }

  template <typename PT, typename RSoA>
  void computeDistances(const PT &pos, const RSoA &R0, T *restrict temp_r,
                        RSoA &temp_dr, int first, int last, int flip_ind = 0)
  {
    const T x0 = pos[0];
    const T y0 = pos[1];
    const T z0 = pos[2];

    const T *restrict px = R0.data(0);
    const T *restrict py = R0.data(1);
    const T *restrict pz = R0.data(2);

    T *restrict dx = temp_dr.data(0);
    T *restrict dy = temp_dr.data(1);
    T *restrict dz = temp_dr.data(2);

    const T *restrict cellx = corners.data(0);
    ASSUME_ALIGNED(cellx);
    const T *restrict celly = corners.data(1);
    ASSUME_ALIGNED(celly);
    const T *restrict cellz = corners.data(2);
    ASSUME_ALIGNED(cellz);

    CONSTEXPR T minusone(-1);
    CONSTEXPR T one(1);
#pragma omp simd aligned(temp_r, px, py, pz, dx, dy, dz)
    for (int iat = first; iat < last; ++iat)
    {
      const T flip    = iat < flip_ind ? one : minusone;
      const T displ_0 = (px[iat] - x0) * flip;
      const T displ_1 = (py[iat] - y0) * flip;
      const T displ_2 = (pz[iat] - z0) * flip;

      const T ar_0 = -std::floor(displ_0 * g00 + displ_1 * g10 + displ_2 * g20);
      const T ar_1 = -std::floor(displ_0 * g01 + displ_1 * g11 + displ_2 * g21);
      const T ar_2 = -std::floor(displ_0 * g02 + displ_1 * g12 + displ_2 * g22);

      const T delx = displ_0 + ar_0 * r00 + ar_1 * r10 + ar_2 * r20;
      const T dely = displ_1 + ar_0 * r01 + ar_1 * r11 + ar_2 * r21;
      const T delz = displ_2 + ar_0 * r02 + ar_1 * r12 + ar_2 * r22;

      T rmin = delx * delx + dely * dely + delz * delz;
      int ic = 0;
#pragma unroll(7)
      for (int c = 1; c < 8; ++c)
      {
        const T x  = delx + cellx[c];
        const T y  = dely + celly[c];
        const T z  = delz + cellz[c];
        const T r2 = x * x + y * y + z * z;
        ic         = (r2 < rmin) ? c : ic;
        rmin       = (r2 < rmin) ? r2 : rmin;
      }

      temp_r[iat] = std::sqrt(rmin);
      dx[iat]     = flip * (delx + cellx[ic]);
      dy[iat]     = flip * (dely + celly[ic]);
      dz[iat]     = flip * (delz + cellz[ic]);
    }
  }
};
}

#endif // OHMMS_PARTICLE_BCONDS_3D_H

/***************************************************************************
 * $RCSfile$   $Author: jeongnim.kim $
 * $Revision: 5204 $   $Date: 2011-04-29 11:33:50 -0500 (Fri, 29 Apr 2011) $
 * $Id: ParticleBConds.h 5204 2011-04-29 16:33:50Z jeongnim.kim $
 ***************************************************************************/
