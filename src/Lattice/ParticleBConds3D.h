////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by:
// Jeremy McMinnis, jmcminis@gmail.com,
//    University of Illinois at Urbana-Champaign
// Jeongnim Kim, jeongnim.kim@gmail.com,
//    University of Illinois at Urbana-Champaign
// Ye Luo, yeluo@anl.gov,
//    Argonne National Laboratory
//
// File created by:
// Jeongnim Kim, jeongnim.kim@gmail.com,
//    University of Illinois at Urbana-Champaign
////////////////////////////////////////////////////////////////////////////////

#ifndef QMCPLUSPLUS_PARTICLE_BCONDS_3D_H
#define QMCPLUSPLUS_PARTICLE_BCONDS_3D_H

#include <Lattice/CrystalLattice.h>
#include <config.h>

namespace qmcplusplus
{

/** specialization for a periodic 3D general cell
 *
 * Wigner-Seitz cell radius > simulation cell radius
 * Need to check image cells
*/
template <class T> struct DTD_BConds<T, 3, PPPG>
{
  T g00, g10, g20, g01, g11, g21, g02, g12, g22;
  TinyVector<TinyVector<T, 3>, 3> rb;
  std::vector<TinyVector<T, 3>> corners;

  DTD_BConds(const CrystalLattice<T, 3> &lat)
  {
    rb[0] = lat.a(0);
    rb[1] = lat.a(1);
    rb[2] = lat.a(2);
    find_reduced_basis(rb);
    Tensor<T, 3> rbt;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) rbt(i, j) = rb[i][j];
    Tensor<T, 3> g                          = inverse(rbt);
    T minusone = -1.0;
    corners.resize(8);
    corners[0] = 0.0;
    corners[1] = minusone * (rb[0]);
    corners[2] = minusone * (rb[1]);
    corners[3] = minusone * (rb[2]);
    corners[4] = minusone * (rb[0] + rb[1]);
    corners[5] = minusone * (rb[0] + rb[2]);
    corners[6] = minusone * (rb[1] + rb[2]);
    corners[7] = minusone * (rb[0] + rb[1] + rb[2]);

    g00 = g(0);
    g10 = g(3);
    g20 = g(6);
    g01 = g(1);
    g11 = g(4);
    g21 = g(7);
    g02 = g(2);
    g12 = g(5);
    g22 = g(8);
  }

  /** apply BC to a displacement vector a and return the minimum-image distance
   * @param lat lattice
   * @param a displacement vector
   * @return the minimum-image distance
   */
  inline T apply_bc(TinyVector<T, 3> &displ) const
  {
    // cart2unit
    TinyVector<T, 3> ar(displ[0] * g00 + displ[1] * g10 + displ[2] * g20,
                        displ[0] * g01 + displ[1] * g11 + displ[2] * g21,
                        displ[0] * g02 + displ[1] * g12 + displ[2] * g22);
    ar[0] = -std::floor(ar[0]);
    ar[1] = -std::floor(ar[1]);
    ar[2] = -std::floor(ar[2]);
    displ += ar[0] * rb[0] + ar[1] * rb[1] + ar[2] * rb[2];
    T rmin2  = dot(displ, displ);
    int imin = 0;
    for (int i = 1; i < corners.size(); ++i)
    {
      TinyVector<T, 3> tv = displ + corners[i];
      T r2 = dot(tv, tv);
      if (r2 < rmin2)
      {
        rmin2 = r2;
        imin  = i;
      }
    }
    if (imin > 0) displ += corners[imin];
    return rmin2;
  }

  inline void apply_bc(std::vector<TinyVector<T, 3>> &dr, std::vector<T> &r,
                       std::vector<T> &rinv) const
  {
    const int n = dr.size();
    const T cone(1);
    for (int i = 0; i < n; ++i)
    {
      r[i]    = std::sqrt(apply_bc(dr[i]));
      rinv[i] = cone / r[i];
    }
  }
};
}

#endif // OHMMS_PARTICLE_BCONDS_3D_H
