/////////////////////////////////////////////////////////////////////////////
//  einspline:  a library for creating and evaluating B-splines            //
//  Copyright (C) 2007 Kenneth P. Esler, Jr.                               //
//                                                                         //
//  This program is free software; you can redistribute it and/or modify   //
//  it under the terms of the GNU General Public License as published by   //
//  the Free Software Foundation; either version 2 of the License, or      //
//  (at your option) any later version.                                    //
//                                                                         //
//  This program is distributed in the hope that it will be useful,        //
//  but WITHOUT ANY WARRANTY; without even the implied warranty of         //
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the          //
//  GNU General Public License for more details.                           //
//                                                                         //
//  You should have received a copy of the GNU General Public License      //
//  along with this program; if not, write to the Free Software            //
//  Foundation, Inc., 51 Franklin Street, Fifth Floor,                     //
//  Boston, MA  02110-1301  USA                                            //
/////////////////////////////////////////////////////////////////////////////
/** @file einspline_allocator.h
 *
 * Rename aligned_alloc/aligned_free as einspline_alloc/einspline_free to
 * avoid naming conflicts with the standards
 */

#ifndef EINSPLINE_ALIGNED_ALLOC_H
#define EINSPLINE_ALIGNED_ALLOC_H

#include <cstddef>
#include "Devices.h"
#include "Numerics/Einspline/multi_bspline_structs.h"
#include "Numerics/Einspline/bspline_structs.h"

void* einspline_alloc(size_t size, size_t alignment);

void einspline_free(void* ptr);


template<Devices D>
void einspline_create_multi_UBspline_3d_s(multi_UBspline_3d_s<D>*& spline, Ugrid x_grid,
                                          Ugrid y_grid, Ugrid z_grid, BCtype_s xBC, BCtype_s yBC,
                                          BCtype_s zBC, int num_splines);

template<Devices D>
void einspline_create_multi_UBspline_3d_d(multi_UBspline_3d_d<D>*& spline, Ugrid x_grid,
                                          Ugrid y_grid, Ugrid z_grid, BCtype_d xBC, BCtype_d yBC,
                                          BCtype_d zBC, int num_splines);


template<Devices D>
void einspline_create_UBspline_3d_s(UBspline_3d_s<D>*& spline, Ugrid x_grid, Ugrid y_grid, Ugrid z_grid,
				    BCtype_s xBC, BCtype_s yBC, BCtype_s zBC);

template<Devices D>
void einspline_create_UBspline_3d_d(UBspline_3d_d<D>*& spline, Ugrid x_grid, Ugrid y_grid, Ugrid z_grid,
				    BCtype_d xBC, BCtype_d yBC, BCtype_d zBC);


//Requires specialization for some devices, specializations should be in device specifc
//cpp files
template<Devices D>
void einspline_create_multi_UBspline_3d_s_coefs(
    multi_UBspline_3d_s<D>*& restrict spline, int Nx, int Ny, int Nz, int N)
{
  spline->coefs = (float*)einspline_alloc(sizeof(float) * spline->coefs_size, QMC_CLINE);
}

//Can require specialization, specializations should be in device specific
//cpp files
template<Devices D>
void einspline_create_multi_UBspline_3d_d_coefs(
    multi_UBspline_3d_d<D>*& restrict spline, int Nx, int Ny, int Nz, int N)
{
  spline->coefs = (double*)einspline_alloc(sizeof(double) * spline->coefs_size, QMC_CLINE);
}

//Can require specialization, specializations should be in device specifc
//cpp files
template<Devices D>
void einspline_create_UBspline_3d_s_coefs(UBspline_3d_s<D>* restrict spline, int Nx, int Ny, int Nz)
{
  spline->coefs = (float*)einspline_alloc(sizeof(float) * spline->coefs_size, QMC_CLINE);
}

//Can require specialization, specializations should be in device specifc
//cpp files
template<Devices D>
void einspline_create_UBspline_3d_d_coefs(UBspline_3d_d<D>* restrict spline, int Nx, int Ny, int Nz)
{
  spline->coefs = (double*)einspline_alloc(sizeof(double) * spline->coefs_size, QMC_CLINE);
}


// This is necessary boilerplate to avoid unecessary multiple compilation.
// Looking into metaprogramming solution or refactor that eleminates the need
// for so much explicit instantiation

extern template void
einspline_create_multi_UBspline_3d_s(multi_UBspline_3d_s<Devices::CPU>*& restrict spline,
                                     Ugrid x_grid, Ugrid y_grid, Ugrid z_grid, BCtype_s xBC,
                                     BCtype_s yBC, BCtype_s zBC, int num_splines);

extern template void einspline_create_multi_UBspline_3d_d(multi_UBspline_3d_d<Devices::CPU>*& spline,
                                                          Ugrid x_grid, Ugrid y_grid, Ugrid z_grid,
                                                          BCtype_d xBC, BCtype_d yBC, BCtype_d zBC,
                                                          int num_splines);


extern template void einspline_create_UBspline_3d_d(UBspline_3d_d<Devices::
						    CPU>*& spline,
                                                    Ugrid x_grid, Ugrid y_grid, Ugrid z_grid,
                                                    BCtype_d xBC, BCtype_d yBC, BCtype_d zBC);

extern template void einspline_create_UBspline_3d_s(UBspline_3d_s<Devices::CPU>*& spline,
                                                    Ugrid x_grid, Ugrid y_grid, Ugrid z_grid,
                                                    BCtype_s xBC, BCtype_s yBC, BCtype_s zBC);

#ifdef QMC_USE_KOKKOS
#include "Numerics/Spline2/einspline_allocator_kokkos.h"
#endif

#endif
