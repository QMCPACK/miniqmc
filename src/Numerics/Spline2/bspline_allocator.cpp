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
#include "Numerics/Spline2/bspline_allocator.hpp"
#include "Numerics/Spline2/einspline_allocator.h"




multi_UBspline_3d_s *
einspline_create_multi_UBspline_3d_s(Ugrid x_grid, Ugrid y_grid, Ugrid z_grid,
                                     BCtype_s xBC, BCtype_s yBC, BCtype_s zBC,
                                     int num_splines);

UBspline_3d_s *einspline_create_UBspline_3d_s(Ugrid x_grid, Ugrid y_grid,
                                              Ugrid z_grid, BCtype_s xBC,
                                              BCtype_s yBC, BCtype_s zBC,
                                              float *data);

multi_UBspline_3d_d *
einspline_create_multi_UBspline_3d_d(Ugrid x_grid, Ugrid y_grid, Ugrid z_grid,
                                     BCtype_d xBC, BCtype_d yBC, BCtype_d zBC,
                                     int num_splines);

UBspline_3d_d *einspline_create_UBspline_3d_d(Ugrid x_grid, Ugrid y_grid,
                                              Ugrid z_grid, BCtype_d xBC,
                                              BCtype_d yBC, BCtype_d zBC,
                                              double *data);

namespace qmcplusplus
{
namespace einspline
{

Allocator::Allocator() : Policy(0) {}

Allocator::~Allocator() {}

multi_UBspline_3d_s *Allocator::allocateMultiBspline(Ugrid x_grid, Ugrid y_grid,
                                                     Ugrid z_grid, BCtype_s xBC,
                                                     BCtype_s yBC, BCtype_s zBC,
                                                     int num_splines)
{
  return einspline_create_multi_UBspline_3d_s(x_grid, y_grid, z_grid, xBC, yBC,
                                              zBC, num_splines);
}

multi_UBspline_3d_d *Allocator::allocateMultiBspline(Ugrid x_grid, Ugrid y_grid,
                                                     Ugrid z_grid, BCtype_d xBC,
                                                     BCtype_d yBC, BCtype_d zBC,
                                                     int num_splines)
{
  return einspline_create_multi_UBspline_3d_d(x_grid, y_grid, z_grid, xBC, yBC,
                                              zBC, num_splines);
}

UBspline_3d_d *Allocator::allocateUBspline(Ugrid x_grid, Ugrid y_grid,
                                           Ugrid z_grid, BCtype_d xBC,
                                           BCtype_d yBC, BCtype_d zBC,
                                           double *data)
{
  return einspline_create_UBspline_3d_d(x_grid, y_grid, z_grid, xBC, yBC, zBC,
                                        data);
}

UBspline_3d_s *Allocator::allocateUBspline(Ugrid x_grid, Ugrid y_grid,
                                           Ugrid z_grid, BCtype_s xBC,
                                           BCtype_s yBC, BCtype_s zBC,
                                           float *data)
{
  return einspline_create_UBspline_3d_s(x_grid, y_grid, z_grid, xBC, yBC, zBC,
                                        data);
}



}
}
