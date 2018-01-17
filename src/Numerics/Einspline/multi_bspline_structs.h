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

#ifndef MULTI_BSPLINE_STRUCTS_STD_H
#define MULTI_BSPLINE_STRUCTS_STD_H

#include <inttypes.h>
#include <stdlib.h>

///////////////////////////
// Single precision real //
///////////////////////////
typedef struct
{
  spline_code spcode;
  type_code tcode;
  typedef Kokkos::View<float**, Kokkos::LayoutRight> coefs_view_t;
  coefs_view_t coefs_view;
  float *QMC_RESTRICT coefs;
  intptr_t x_stride;
  Ugrid x_grid;
  BCtype_s xBC;
  int num_splines;
  size_t coefs_size;
} multi_UBspline_1d_s;

typedef struct
{
  spline_code spcode;
  type_code tcode;
  typedef Kokkos::View<float***, Kokkos::LayoutRight> coefs_view_t;
  coefs_view_t coefs_view;
  float *QMC_RESTRICT coefs;
  intptr_t x_stride, y_stride;
  Ugrid x_grid, y_grid;
  BCtype_s xBC, yBC;
  int num_splines;
} multi_UBspline_2d_s;

typedef struct
{
  spline_code spcode;
  type_code tcode;
  typedef Kokkos::View<float****, Kokkos::LayoutRight> coefs_view_t;
  coefs_view_t coefs_view;
  float *QMC_RESTRICT coefs;
  intptr_t x_stride, y_stride, z_stride;
  Ugrid x_grid, y_grid, z_grid;
  BCtype_s xBC, yBC, zBC;
  int num_splines;
  size_t coefs_size;
} multi_UBspline_3d_s;

///////////////////////////
// Double precision real //
///////////////////////////
typedef struct
{
  spline_code spcode;
  type_code tcode;
  typedef Kokkos::View<double**, Kokkos::LayoutRight> coefs_view_t;
  coefs_view_t coefs_view;
  double *QMC_RESTRICT coefs;
  intptr_t x_stride;
  Ugrid x_grid;
  BCtype_d xBC;
  int num_splines;
  size_t coefs_size;
} multi_UBspline_1d_d;

typedef struct
{
  spline_code spcode;
  type_code tcode;
  typedef Kokkos::View<double***, Kokkos::LayoutRight> coefs_view_t;
  coefs_view_t coefs_view;
  double *QMC_RESTRICT coefs;
  intptr_t x_stride, y_stride;
  Ugrid x_grid, y_grid;
  BCtype_d xBC, yBC;
  int num_splines;
} multi_UBspline_2d_d;

typedef struct
{
  spline_code spcode;
  type_code tcode;
  typedef Kokkos::View<double****, Kokkos::LayoutRight> coefs_view_t;
  coefs_view_t coefs_view;
  double *QMC_RESTRICT coefs;
  intptr_t x_stride, y_stride, z_stride;
  Ugrid x_grid, y_grid, z_grid;
  BCtype_d xBC, yBC, zBC;
  int num_splines;
  size_t coefs_size;
} multi_UBspline_3d_d;

//////////////////////////////
// Single precision complex //
//////////////////////////////
typedef struct
{
  spline_code spcode;
  type_code tcode;
  typedef Kokkos::View<complex_float**, Kokkos::LayoutRight> coefs_view_t;
  coefs_view_t coefs_view;
  complex_float *QMC_RESTRICT coefs;
  intptr_t x_stride;
  Ugrid x_grid;
  BCtype_c xBC;
  int num_splines;
  size_t coefs_size;
} multi_UBspline_1d_c;

typedef struct
{
  spline_code spcode;
  type_code tcode;
  typedef Kokkos::View<complex_float***, Kokkos::LayoutRight> coefs_view_t;
  coefs_view_t coefs_view;
  complex_float *QMC_RESTRICT coefs;
  intptr_t x_stride, y_stride;
  Ugrid x_grid, y_grid;
  BCtype_c xBC, yBC;
  int num_splines;
  // temporary storage for laplacian components
  complex_float *QMC_RESTRICT lapl2;
} multi_UBspline_2d_c;

typedef struct
{
  spline_code spcode;
  type_code tcode;
  typedef Kokkos::View<complex_float**, Kokkos::LayoutRight> coefs_view_t;
  coefs_view_t coefs_view;
  complex_float *QMC_RESTRICT coefs;
  intptr_t x_stride, y_stride, z_stride;
  Ugrid x_grid, y_grid, z_grid;
  BCtype_c xBC, yBC, zBC;
  int num_splines;
  size_t coefs_size;
  // temporary storage for laplacian components
  complex_float *QMC_RESTRICT lapl3;
} multi_UBspline_3d_c;

//////////////////////////////
// Double precision complex //
//////////////////////////////
typedef struct
{
  spline_code spcode;
  type_code tcode;
  typedef Kokkos::View<complex_double**, Kokkos::LayoutRight> coefs_view_t;
  coefs_view_t coefs_view;
  complex_double *QMC_RESTRICT coefs;
  intptr_t x_stride;
  Ugrid x_grid;
  BCtype_z xBC;
  int num_splines;
  size_t coefs_size;
} multi_UBspline_1d_z;

typedef struct
{
  spline_code spcode;
  type_code tcode;
  typedef Kokkos::View<complex_double***, Kokkos::LayoutRight> coefs_view_t;
  coefs_view_t coefs_view;
  complex_double *QMC_RESTRICT coefs;
  intptr_t x_stride, y_stride;
  Ugrid x_grid, y_grid;
  BCtype_z xBC, yBC;
  int num_splines;
  // temporary storage for laplacian components
  complex_double *QMC_RESTRICT lapl2;
} multi_UBspline_2d_z;

typedef struct
{
  spline_code spcode;
  type_code tcode;
  typedef Kokkos::View<complex_double****, Kokkos::LayoutRight> coefs_view_t;
  coefs_view_t coefs_view;
  complex_double *QMC_RESTRICT coefs;
  intptr_t x_stride, y_stride, z_stride;
  Ugrid x_grid, y_grid, z_grid;
  BCtype_z xBC, yBC, zBC;
  int num_splines;
  size_t coefs_size;
  // temporary storage for laplacian components
  complex_double *QMC_RESTRICT lapl3;
} multi_UBspline_3d_z;

#endif
