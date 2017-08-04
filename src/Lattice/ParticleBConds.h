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
//
// File created by:
// Jeongnim Kim, jeongnim.kim@gmail.com,
//    University of Illinois at Urbana-Champaign
////////////////////////////////////////////////////////////////////////////////

#ifndef QMCPLUSPLUS_PARTICLE_BCONDS_H
#define QMCPLUSPLUS_PARTICLE_BCONDS_H

#include <Lattice/CrystalLattice.h>
#include <config.h>

namespace qmcplusplus
{

// clang-format off
/** generic Boundary condition handler
 *
 * @tparam T real data type
 * @tparam D physical dimension
 * @tparm SC supercell type
 *
 * Default method for any dimension with OPEN boundary condition.
 * \htmlonly
 <table>
 <th>
 <td>SC</td><td>3D</td><td>2D</td><td>1D</td><td>comment</td>
 </th>
 <tr><td>SUPERCELL_OPEN           </td><td>n n n</td><td>n n</td><td>n</td><td>open boudary conditions </td> </tr>
 <tr><td>SUPERCELL_BULK           </td><td>p p p</td><td>NA   </td><td>NA</td><td>periodic boundry conditions in 3 dimensions, general cell</td> </tr>
 <tr><td>SUPERCELL_BULK+TwoPowerD </td><td>p p p</td><td>NA   </td><td>NA</td><td>periodic boundry conditions in 3 dimensions, orthorombic cell</td></tr>
 <tr><td>SUPERCELL_SLAB           </td><td>p p n</td><td>p p</td><td>NA</td><td>periodic boundry conditions in 2 dimensions, general cell</td></tr>
 <tr><td>SUPERCELL_SLAB+TwoPowerD </td><td>p p n</td><td>p p</td><td>NA</td><td>periodic boundry conditions in 2 dimensions, orthorombic cell</td></tr>
 <tr><td>SUPERCELL_WIRE           </td><td>p n n</td><td>p n</td><td>p</td><td>periodic boundry conditions in 1 dimension</td></tr>
 </table>
 * \endhtmlonly
 * Specialization of DTD_BConds should implement
 * - apply_bc(TinyVector<T,D>& displ): apply BC on displ, Cartesian displacement vector, and returns |displ|^2
 * - apply_bc(dr,r,rinv): apply BC on displacements
 */
// clang-format on

template <class T, unsigned D, int SC> struct DTD_BConds
{

  /** constructor: doing nothing */
  inline DTD_BConds(const CrystalLattice<T, D> &lat)
  {
    APP_ABORT("qmcplusplus::DTD_BConds default DTD_BConds is not allowed in "
              "miniQMC!\n");
  }

  /** apply BC on displ and return |displ|^2
   * @param displ a displacement vector in the Cartesian coordinate
   * @return \f$|displ|^2\f$
   */
  inline T apply_bc(TinyVector<T, D> &displ) const { return dot(displ, displ); }

  /** apply BC on dr and evaluate r and rinv
   * @param dr vector of displacements, in and out
   * @param r vector of distances
   * @param rinv vector of 1/r
   *
   * The input displacement vectors are not modified with the open boundary
   * conditions.
   */
  inline void apply_bc(std::vector<TinyVector<T, D>> &dr, std::vector<T> &r,
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

#include <Lattice/ParticleBConds3D.h>

#endif // OHMMS_PARTICLE_BCONDS_H
