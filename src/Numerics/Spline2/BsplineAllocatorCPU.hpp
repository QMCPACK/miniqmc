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
#ifndef QMCPLUSPLUS_BSPLINE_ALLOCATOR_CPU_H
#define QMCPLUSPLUS_BSPLINE_ALLOCATOR_CPU_H

#include "Numerics/Spline2/BsplineAllocator.hpp"

namespace qmcplusplus
{
namespace einspline
{
template<>
void Allocator<Devices::CPU>::allocateMultiBspline(multi_UBspline_3d_s<Devices::CPU>*& spline,
                                                   Ugrid x_grid, Ugrid y_grid, Ugrid z_grid,
                                                   BCtype_s xBC, BCtype_s yBC, BCtype_s zBC,
                                                   int num_splines)
{
  einspline_create_multi_UBspline_3d_s(spline, x_grid, y_grid, z_grid, xBC, yBC, zBC, num_splines);
}

template<>
void Allocator<Devices::CPU>::allocateMultiBspline(multi_UBspline_3d_d<Devices::CPU>*& spline,
                                                   Ugrid x_grid, Ugrid y_grid, Ugrid z_grid,
                                                   BCtype_d xBC, BCtype_d yBC, BCtype_d zBC,
                                                   int num_splines)
{
  einspline_create_multi_UBspline_3d_d(spline, x_grid, y_grid, z_grid, xBC, yBC, zBC, num_splines);
}

template<>
void Allocator<Devices::CPU>::allocateUBspline(UBspline_3d_s<Devices::CPU>*& spline, Ugrid x_grid,
                                               Ugrid y_grid, Ugrid z_grid, BCtype_s xBC,
                                               BCtype_s yBC, BCtype_s zBC)
{
  einspline_create_UBspline_3d_s(spline, x_grid, y_grid, z_grid, xBC, yBC, zBC);
}

template<>
void Allocator<Devices::CPU>::allocateUBspline(UBspline_3d_d<Devices::CPU>*& spline, Ugrid x_grid,
                                               Ugrid y_grid, Ugrid z_grid, BCtype_d xBC,
                                               BCtype_d yBC, BCtype_d zBC)
{
  einspline_create_UBspline_3d_d(spline, x_grid, y_grid, z_grid, xBC, yBC, zBC);
}

extern template class Allocator<Devices::CPU>;
extern template void Allocator<Devices::CPU>::destroy(multi_UBspline_3d_d<Devices::CPU>*&);
extern template void Allocator<Devices::CPU>::destroy(multi_UBspline_3d_s<Devices::CPU>*&);

} // namespace einspline
} // namespace qmcplusplus
#endif
