////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2019 QMCPACK developers.
//
// File developed by:
// Peter Doak, doakpw@ornl.gov, Oak Ridge National Lab
//
// File created by:
// Peter Doak, doakpw@ornl.gov, Oak Ridge National Lab
////////////////////////////////////////////////////////////////////////////////
// -*- C++ -*-
/** @file 
 * @class template specialization for KOKKOS
 * only necessary due to weirdness or my lack of understanding of KOKKOS::View<***>
 */
#ifndef QMCPLUSPLUS_BSPLINE_ALLOCATOR_KOKKOS_H
#define QMCPLUSPLUS_BSPLINE_ALLOCATOR_KOKKOS_H

#include "clean_inlining.h"
#include "Devices.h"
#include <Numerics/Spline2/bspline_traits.hpp>

namespace qmcplusplus
{
namespace einspline
{
template<Devices DT>
class Allocator;
    
template<>
class Allocator<Devices::KOKKOS>
{
  static constexpr Devices DT = Devices::KOKKOS;
  /// Setting the allocation policy: default is using aligned allocator
  int Policy;

public:
  /// constructor
  Allocator();
  Allocator(const Allocator&) = default;
  /// disable assignement
  Allocator& operator=(const Allocator&) = delete;
  /// destructor
  ~Allocator();

  template<typename ST>
  void destroy(ST*& spline);

  /// allocate a single multi-bspline
  void allocateMultiBspline(multi_UBspline_3d_s<DT>*& spline,
                            Ugrid x_grid,
                            Ugrid y_grid,
                            Ugrid z_grid,
                            BCtype_s xBC,
                            BCtype_s yBC,
                            BCtype_s zBC,
                            int num_splines);

  /// allocate a double multi-bspline
  void allocateMultiBspline(multi_UBspline_3d_d<DT>*& spline,
                            Ugrid x_grid,
                            Ugrid y_grid,
                            Ugrid z_grid,
                            BCtype_d xBC,
                            BCtype_d yBC,
                            BCtype_d zBC,
                            int num_splines);

  /// allocate a single bspline
  void allocateUBspline(UBspline_3d_s<DT>*& spline, Ugrid x_grid, Ugrid y_grid, Ugrid z_grid,
                        BCtype_s xBC, BCtype_s yBC, BCtype_s zBC);

  /// allocate a UBspline_3d_d
  void allocateUBspline(UBspline_3d_d<DT>*& spline, Ugrid x_grid, Ugrid y_grid, Ugrid z_grid,
                        BCtype_d xBC, BCtype_d yBC, BCtype_d zBC);

  /** allocate a multi_UBspline_3d_(s,d)
   * @tparam T datatype
   * @tparam ValT 3D container for start and end
   * @tparam IntT 3D container for ng
   */
  template<typename T, typename ValT, typename IntT>
  void createMultiBspline(typename bspline_traits<DT, T, 3>::SplineType*& spline, T dummy,
                          ValT& start, ValT& end, IntT& ng, bc_code bc, int num_splines);

  /** allocate a UBspline_3d_(s,d)
   * @tparam T datatype
   * @tparam ValT 3D container for start and end
   * @tparam IntT 3D container for ng
   */
  template<typename ValT, typename IntT, typename T>
  void createUBspline(typename bspline_traits<DT, T, 3>::SingleSplineType*& spline, ValT& start,
                      ValT& end, IntT& ng, bc_code bc);

  /** Set coefficients for a single orbital (band)
   * @param i index of the orbital
   * @param coeff array of coefficients
   * @param spline target MultibsplineType
   */
  template<typename T>
  void setCoefficientsForOneOrbital(int i,
                                    Kokkos::View<T***>&,
                                    typename bspline_traits<DT, T, 3>::SplineType* spline);

  /** copy a UBSpline_3d_X to multi_UBspline_3d_X at i-th band
   * @param single  UBspline_3d_X
   * @param multi target multi_UBspline_3d_X
   * @param i the band index to copy to
   * @param offset starting offset for AoSoA
   * @param N shape of AoSoA
   */
  template<typename UBT, typename MBT>
  void copy(UBT* single, MBT* multi, int i, const int* offset, const int* N);
};
extern template class Allocator<Devices::KOKKOS>;

}
}
#endif
