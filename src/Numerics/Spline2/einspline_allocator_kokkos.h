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
/** @file
 *
 * Rename aligned_alloc/aligned_free as einspline_alloc/einspline_free to
 * avoid naming conflicts with the standards
 */

#ifndef EINSPLINE_ALLOCATOR_KOKKOS_H
#define EINSPLINE_ALLOCATOR_KOKKOS_H

#include "Numerics/Einspline/multi_bspline_structs_kokkos.h"
#include "Numerics/Einspline/bspline_structs_kokkos.h"
#include "Numerics/Spline2/einspline_allocator.h"

template<>
inline void
einspline_create_multi_UBspline_3d_s(multi_UBspline_3d_s<Devices::KOKKOS>*& restrict spline,
                                     Ugrid x_grid, Ugrid y_grid, Ugrid z_grid, BCtype_s xBC,
                                     BCtype_s yBC, BCtype_s zBC, int num_splines);

template<>
inline void
einspline_create_multi_UBspline_3d_d(multi_UBspline_3d_d<Devices::KOKKOS>*& spline, Ugrid x_grid,
                                     Ugrid y_grid, Ugrid z_grid, BCtype_d xBC, BCtype_d yBC,
                                     BCtype_d zBC, int num_splines);


//inlined only due to inclusion in multiple compilation units
template<>
inline void
einspline_create_multi_UBspline_3d_s_coefs(multi_UBspline_3d_s<Devices::KOKKOS>*& restrict spline,
                                           int Nx, int Ny, int Nz, int N);

template<>
inline void
einspline_create_multi_UBspline_3d_d_coefs(multi_UBspline_3d_d<Devices::KOKKOS>*& restrict spline,
                                           int Nx, int Ny, int Nz, int N);

extern template void einspline_create_UBspline_3d_d(UBspline_3d_d<Devices::KOKKOS>*& spline,
                                                    Ugrid x_grid, Ugrid y_grid, Ugrid z_grid,
                                                    BCtype_d xBC, BCtype_d yBC, BCtype_d zBC);

extern template void einspline_create_UBspline_3d_s(UBspline_3d_s<Devices::KOKKOS>*& spline,
                                                    Ugrid x_grid, Ugrid y_grid, Ugrid z_grid,
                                                    BCtype_s xBC, BCtype_s yBC, BCtype_s zBC);


/* extern template void einspline_create_UBspline_3d_d_coefs(UBspline_3d_d<Devices::KOKKOS>*& spline, */
/* 							  int Nx, int Ny, int Nz, int N); */

/* extern template void einspline_create_UBspline_3d_s_coefs(UBspline_3d_s<Devices::KOKKOS>*& spline, */
/* 							  int Nx, int Ny, int Nz, int N); */


#endif
