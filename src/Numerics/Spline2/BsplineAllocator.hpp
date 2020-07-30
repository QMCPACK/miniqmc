//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by:
//
// File created by: Jeongnim Kim, jeongnim.kim@intel.com, Intel Corp.
//////////////////////////////////////////////////////////////////////////////////////
// -*- C++ -*-
/** @file bspline_allocator.hpp
 * @brief BsplineAllocator and management classes
 */
#ifndef QMCPLUSPLUS_EINSPLINE_BSPLINE_ALLOCATOR_H
#define QMCPLUSPLUS_EINSPLINE_BSPLINE_ALLOCATOR_H

#include "Utilities/SIMD/Mallocator.hpp"
#include <cmath>
#include "Numerics/Spline2/bspline_traits.hpp"
#include <Numerics/OhmmsPETE/OhmmsArray.h>

namespace qmcplusplus
{
template<typename T, size_t ALIGN = QMC_CLINE, typename ALLOC = Mallocator<T, ALIGN>>
class BsplineAllocator
{
  using SplineType       = typename bspline_traits<T, 3>::SplineType;
  using SingleSplineType = typename bspline_traits<T, 3>::SingleSplineType;
  using BCType           = typename bspline_traits<T, 3>::BCType;
  using real_type        = typename bspline_traits<T, 3>::real_type;

  /// allocator
  ALLOC mAllocator;

public:
  ///default constructor
  BsplineAllocator() = default;
  ///default destructor
  ~BsplineAllocator() = default;
  ///disable copy constructor
  BsplineAllocator(const BsplineAllocator&) = delete;
  ///disable assignement
  BsplineAllocator& operator=(const BsplineAllocator&) = delete;

  template<typename SplineType>
  void destroy(SplineType* spline)
  {
    mAllocator.deallocate(spline->coefs, spline->coefs_size);
    delete (spline);
  }

  ///allocate a multi-bspline structure
  SplineType* allocateMultiBspline(
      Ugrid x_grid, Ugrid y_grid, Ugrid z_grid, BCType xBC, BCType yBC, BCType zBC, int num_splines);

  /** allocate a multi_UBspline_3d_(s,d)
   * @tparam T datatype
   * @tparam ValT 3D container for start and end
   * @tparam IntT 3D container for ng
   */
  template<typename ValT, typename IntT>
  typename bspline_traits<T, 3>::SplineType*
  createMultiBspline(T dummy, ValT& start, ValT& end, IntT& ng, bc_code bc, int num_splines);

  /** Set coefficients for a single orbital (band)
   * @param i index of the orbital
   * @param coeff array of coefficients
   * @param spline target MultibsplineType
   */
  void setCoefficientsForOrbitals(int first, int last, Array<T, 3>& coeff, SplineType* spline);

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

template<typename T, size_t ALIGN, typename ALLOC>
typename BsplineAllocator<T, ALIGN, ALLOC>::SplineType*
BsplineAllocator<T, ALIGN, ALLOC>::allocateMultiBspline(
    Ugrid x_grid, Ugrid y_grid, Ugrid z_grid, BCType xBC, BCType yBC, BCType zBC, int num_splines)
{
  // Create new spline
  SplineType* restrict spline = new SplineType;
  spline->xBC                 = xBC;
  spline->yBC                 = yBC;
  spline->zBC                 = zBC;
  spline->num_splines         = num_splines;

  // Setup internal variables
  int Mx = x_grid.num;
  int My = y_grid.num;
  int Mz = z_grid.num;
  int Nx, Ny, Nz;

  if (xBC.lCode == PERIODIC || xBC.lCode == ANTIPERIODIC)
    Nx = Mx + 3;
  else
    Nx = Mx + 2;
  x_grid.delta     = (x_grid.end - x_grid.start) / (double)(Nx - 3);
  x_grid.delta_inv = 1.0 / x_grid.delta;
  spline->x_grid   = x_grid;

  if (yBC.lCode == PERIODIC || yBC.lCode == ANTIPERIODIC)
    Ny = My + 3;
  else
    Ny = My + 2;
  y_grid.delta     = (y_grid.end - y_grid.start) / (double)(Ny - 3);
  y_grid.delta_inv = 1.0 / y_grid.delta;
  spline->y_grid   = y_grid;

  if (zBC.lCode == PERIODIC || zBC.lCode == ANTIPERIODIC)
    Nz = Mz + 3;
  else
    Nz = Mz + 2;
  z_grid.delta     = (z_grid.end - z_grid.start) / (double)(Nz - 3);
  z_grid.delta_inv = 1.0 / z_grid.delta;
  spline->z_grid   = z_grid;

  const int N = getAlignedSize<real_type, ALIGN>(num_splines);

  spline->x_stride = (size_t)Ny * (size_t)Nz * (size_t)N;
  spline->y_stride = Nz * N;
  spline->z_stride = N;

  spline->coefs_size = (size_t)Nx * spline->x_stride;
  spline->coefs      = mAllocator.allocate(spline->coefs_size);

  return spline;
}

template<typename T, size_t ALIGN, typename ALLOC>
template<typename ValT, typename IntT>
typename bspline_traits<T, 3>::SplineType* BsplineAllocator<T, ALIGN, ALLOC>::createMultiBspline(
    T dummy, ValT& start, ValT& end, IntT& ng, bc_code bc, int num_splines)
{
  Ugrid x_grid, y_grid, z_grid;
  typename bspline_traits<T, 3>::BCType xBC, yBC, zBC;
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
  return allocateMultiBspline(x_grid, y_grid, z_grid, xBC, yBC, zBC, num_splines);
}

template<typename T, size_t ALIGN, typename ALLOC>
void BsplineAllocator<T, ALIGN, ALLOC>::setCoefficientsForOrbitals(int first,
                                                                   int last,
                                                                   Array<T, 3>& coeff,
                                                                   SplineType* spline)
{
  const int size = last - first;
  std::vector<T> prefactor(size);
  for (int ind = first; ind < last; ind++)
    prefactor[ind] = std::cos(2 * M_PI * ind / size);

#pragma omp parallel for collapse(3)
  for (int ix = 0; ix < spline->x_grid.num + 3; ix++)
    for (int iy = 0; iy < spline->y_grid.num + 3; iy++)
      for (int iz = 0; iz < spline->z_grid.num + 3; iz++)
      {
        intptr_t xs = spline->x_stride;
        intptr_t ys = spline->y_stride;
        intptr_t zs = spline->z_stride;
        for (int ind = first; ind < last; ind++)
          spline->coefs[ix * xs + iy * ys + iz * zs + ind] = coeff(ix, iy, iz) * prefactor[ind];
      }
}

template<typename T, size_t ALIGN, typename ALLOC>
template<typename UBT, typename MBT>
void BsplineAllocator<T, ALIGN, ALLOC>::copy(
    UBT* single, MBT* multi, int i, const int* offset, const int* N)
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

} // namespace qmcplusplus
#endif
