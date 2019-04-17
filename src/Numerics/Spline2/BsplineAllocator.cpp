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
/** @file bspline_allocator.cpp
 * @brief Implementation of einspline::Allocator member functions
 *
 * Allocator::Policy is not defined precisely yet but is intended to select
 * the specialized allocator.
 */
#include "Devices.h"
#include "Numerics/Spline2/BsplineAllocator.hpp"
#include "Numerics/Spline2/BsplineAllocatorCPU.hpp"
#include "Numerics/Spline2/einspline_allocator.h"


template<Devices DT>
void einspline_create_multi_UBspline_3d_s(multi_UBspline_3d_s<DT>* spline,
                                          Ugrid x_grid,
                                          Ugrid y_grid,
                                          Ugrid z_grid,
                                          BCtype_s xBC,
                                          BCtype_s yBC,
                                          BCtype_s zBC,
                                          int num_splines);

template<Devices DT>
void einspline_create_UBspline_3d_s(UBspline_3d_s<DT>* spline, Ugrid x_grid, Ugrid y_grid,
                                    Ugrid z_grid, BCtype_s xBC, BCtype_s yBC, BCtype_s zBC);

template<Devices DT>
void einspline_create_multi_UBspline_3d_d(multi_UBspline_3d_d<DT>* spline,
                                          Ugrid x_grid,
                                          Ugrid y_grid,
                                          Ugrid z_grid,
                                          BCtype_d xBC,
                                          BCtype_d yBC,
                                          BCtype_d zBC,
                                          int num_splines);

template<Devices DT>
void einspline_create_UBspline_3d_d(UBspline_3d_d<DT>* spline, Ugrid x_grid, Ugrid y_grid,
                                    Ugrid z_grid, BCtype_d xBC, BCtype_d yBC, BCtype_d zBC);

namespace qmcplusplus
{
namespace einspline
{
template<Devices DT>
Allocator<DT>::Allocator() : Policy(0)
{}

template<Devices DT>
Allocator<DT>::~Allocator()
{}

template<>
template<typename SplineType>
void Allocator<Devices::CPU>::destroy(SplineType*& spline)
{
  einspline_free(spline->coefs);
  free(spline);
}

template class Allocator<Devices::CPU>;
template void Allocator<Devices::CPU>::destroy(multi_UBspline_3d_d<Devices::CPU>*&);
template void Allocator<Devices::CPU>::destroy(multi_UBspline_3d_s<Devices::CPU>*&);

} // namespace einspline
} // namespace qmcplusplus
