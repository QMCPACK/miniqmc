////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by:
//
// File created by: Jeongnim Kim, jeongnim.kim@intel.com, Intel Corp.
////////////////////////////////////////////////////////////////////////////////
// -*- C++ -*-
/** @file bspline_allocator.hpp
 * @brief Allocator and management classes
 */
#ifndef QMCPLUSPLUS_EINSPLINE_BSPLINE_ALLOCATOR_H
#define QMCPLUSPLUS_EINSPLINE_BSPLINE_ALLOCATOR_H

#include "Devices.h"
#include <Utilities/SIMD/allocator.hpp>
#include <Numerics/Spline2/bspline_traits.hpp>
#include "Numerics/Spline2/einspline_allocator.h"
#include <Numerics/OhmmsPETE/OhmmsArray.h>
#ifdef QMC_USE_KOKKOS
#include "Numerics/Spline2/bspline_allocator_KOKKOS.hpp"
#endif
namespace qmcplusplus
{
namespace einspline
{
using qmcplusplus::Devices;
template<Devices D>
class Allocator
{
  /// Setting the allocation policy: default is using aligned allocator
  int Policy;

public:
  /// constructor
  Allocator();
#if (__cplusplus >= 201103L)
  /// enable default copy constructor
  Allocator(const Allocator&) = default;
  /// disable assignement
  Allocator& operator=(const Allocator&) = delete;
#endif
  /// destructor
  ~Allocator();

  template<typename SplineType>
  void destroy(SplineType* spline);

  /// allocate a single multi-bspline
  void allocateMultiBspline(multi_UBspline_3d_s<D>*& spline,
					       Ugrid x_grid,
                                            Ugrid y_grid,
                                            Ugrid z_grid,
                                            BCtype_s xBC,
                                            BCtype_s yBC,
                                            BCtype_s zBC,
                                            int num_splines);

  /// allocate a double multi-bspline
  void allocateMultiBspline(multi_UBspline_3d_d<D>*& spline,
			    Ugrid x_grid,
                                            Ugrid y_grid,
                                            Ugrid z_grid,
                                            BCtype_d xBC,
                                            BCtype_d yBC,
                                            BCtype_d zBC,
                                            int num_splines);

  /// allocate a single bspline
  void
  allocateUBspline(UBspline_3d_s<D>*& spline, Ugrid x_grid, Ugrid y_grid, Ugrid z_grid, BCtype_s xBC, BCtype_s yBC, BCtype_s zBC);

  /// allocate a UBspline_3d_d
  void
  allocateUBspline(UBspline_3d_d<D>*& spline, Ugrid x_grid, Ugrid y_grid, Ugrid z_grid, BCtype_d xBC, BCtype_d yBC, BCtype_d zBC);

  /** allocate a multi_UBspline_3d_(s,d)
   * @tparam T datatype
   * @tparam ValT 3D container for start and end
   * @tparam IntT 3D container for ng
   */
  template<typename T, typename ValT, typename IntT>
  void createMultiBspline(typename bspline_traits<D, T, 3>::SplineType*& spline,
		     T dummy, ValT& start, ValT& end, IntT& ng, bc_code bc, int num_splines);

  /** allocate a UBspline_3d_(s,d)
   * @tparam T datatype
   * @tparam ValT 3D container for start and end
   * @tparam IntT 3D container for ng
   */
  template<typename ValT, typename IntT, typename T>
  void createUBspline(typename bspline_traits<D, T, 3>::SingleSplineType*& spline,
		      ValT& start, ValT& end, IntT& ng, bc_code bc);

   /** Set coefficients for a single orbital (band)
   * @param i index of the orbital
   * @param coeff array of coefficients
   * @param spline target MultibsplineType
   */
  template<typename T>
  void setCoefficientsForOneOrbital(int i,
                                    Array<T, 3>& coeff,
                                    typename bspline_traits<D, T, 3>::SplineType* spline);

#ifdef QMC_USE_KOKKOS
  /** Set coefficients for a single orbital (band)
   * @param i index of the orbital
   * @param coeff array of coefficients
   * @param spline target MultibsplineType
   */
  template<typename T>
  void setCoefficientsForOneOrbital(int i,
                                    Kokkos::View<T***>& coeff,

                                    typename bspline_traits<D, T, 3>::SplineType* spline);
#endif

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

template<Devices D>
template<typename T>
void Allocator<D>::setCoefficientsForOneOrbital(int i,
                                             Array<T, 3>& coeff,
						typename bspline_traits<D, T, 3>::SplineType* spline)
{
  //#pragma omp parallel for collapse(3)
  for (int ix = 0; ix < spline->x_grid.num + 3; ix++)
  {
    for (int iy = 0; iy < spline->y_grid.num + 3; iy++)
    {
      for (int iz = 0; iz < spline->z_grid.num + 3; iz++)
      {
        intptr_t xs                                    = spline->x_stride;
        intptr_t ys                                    = spline->y_stride;
        intptr_t zs                                    = spline->z_stride;
        spline->coefs[ix * xs + iy * ys + iz * zs + i] = coeff(ix, iy, iz);
      }
    }
  }
}

#ifdef QMC_USE_KOKKOS
template<Devices D>
template<typename T>
void Allocator<D>::setCoefficientsForOneOrbital(int i,
                                             Kokkos::View<T***>& coeff,
						typename bspline_traits<D, T, 3>::SplineType* spline)
{
  //  #pragma omp parallel for collapse(3)
  for (int ix = 0; ix < spline->x_grid.num + 3; ix++)
  {
    for (int iy = 0; iy < spline->y_grid.num + 3; iy++)
    {
      for (int iz = 0; iz < spline->z_grid.num + 3; iz++)
      {
        intptr_t xs                                    = spline->x_stride;
        intptr_t ys                                    = spline->y_stride;
        intptr_t zs                                    = spline->z_stride;
        spline->coefs[ix * xs + iy * ys + iz * zs + i] = coeff(ix, iy, iz);
      }
    }
  }
}
#endif

template<Devices D>
template<typename T, typename ValT, typename IntT>
void Allocator<D>::createMultiBspline(typename bspline_traits<D, T, 3>::SplineType*& spline,
				 T dummy, ValT& start, ValT& end, IntT& ng, bc_code bc, int num_splines)
{
  Ugrid x_grid, y_grid, z_grid;
  typename bspline_traits<D, T, 3>::BCType xBC, yBC, zBC;
  x_grid.start = start[0];
  x_grid.end   = end[0];
  x_grid.num   = ng[0];
  y_grid.start = start[1];
  y_grid.end   = end[1];
  y_grid.num   = ng[1];
  z_grid.start = start[2];
  z_grid.end   = end[2];
  z_grid.num   = ng[2];
  xBC.lCode = xBC.rCode = bc;
  yBC.lCode = yBC.rCode = bc;
  zBC.lCode = zBC.rCode = bc;
  allocateMultiBspline(spline, x_grid, y_grid, z_grid, xBC, yBC, zBC, num_splines);
}

template<Devices D>
template<typename ValT, typename IntT, typename T>
void Allocator<D>::createUBspline(typename bspline_traits<D, T, 3>::SingleSplineType*& spline,
			       ValT& start, ValT& end, IntT& ng, bc_code bc)
{
  Ugrid x_grid, y_grid, z_grid;
  typename bspline_traits<D, T, 3>::BCType xBC, yBC, zBC;
  x_grid.start = start[0];
  x_grid.end   = end[0];
  x_grid.num   = ng[0];
  y_grid.start = start[1];
  y_grid.end   = end[1];
  y_grid.num   = ng[1];
  z_grid.start = start[2];
  z_grid.end   = end[2];
  z_grid.num   = ng[2];
  xBC.lCode = xBC.rCode = bc;
  yBC.lCode = yBC.rCode = bc;
  zBC.lCode = zBC.rCode = bc;
  allocateUBspline(spline, x_grid, y_grid, z_grid, xBC, yBC, zBC);
}

template<Devices D>
template<typename UBT, typename MBT>
void Allocator<D>::copy(UBT* single, MBT* multi, int i, const int* offset, const int* N)
{
  typedef typename bspline_type<MBT>::value_type out_type;
  typedef typename bspline_type<UBT>::value_type in_type;
  intptr_t x_stride_in  = single->x_stride;
  intptr_t y_stride_in  = single->y_stride;
  intptr_t x_stride_out = multi->x_stride;
  intptr_t y_stride_out = multi->y_stride;
  intptr_t z_stride_out = multi->z_stride;
  intptr_t offset0      = static_cast<intptr_t>(offset[0]);
  intptr_t offset1      = static_cast<intptr_t>(offset[1]);
  intptr_t offset2      = static_cast<intptr_t>(offset[2]);
  const intptr_t istart = static_cast<intptr_t>(i);
  const intptr_t n0 = N[0], n1 = N[1], n2 = N[2];
  for (intptr_t ix = 0; ix < n0; ++ix)
    for (intptr_t iy = 0; iy < n1; ++iy)
    {
      out_type* restrict out = multi->coefs + ix * x_stride_out + iy * y_stride_out + istart;
      const in_type* restrict in =
          single->coefs + (ix + offset0) * x_stride_in + (iy + offset1) * y_stride_in + offset2;
      for (intptr_t iz = 0; iz < n2; ++iz)
      {
        out[iz * z_stride_out] = static_cast<out_type>(in[iz]);
      }
    }
}

extern template class Allocator<Devices::CPU>;
extern template void Allocator<Devices::CPU>::destroy(multi_UBspline_3d_s<Devices::CPU>*);
extern template void Allocator<Devices::CPU>::destroy(multi_UBspline_3d_d<Devices::CPU>*);

#ifdef QMC_USE_KOKKOS
extern template class Allocator<Devices::KOKKOS>;
#endif
#ifdef QMC_USE_CUDA
extern template class Allocator<Devices::CUDA>;
#endif

} // namespace einspline
} // namespace qmcplusplus
#endif
