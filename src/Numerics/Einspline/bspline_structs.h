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

#ifndef BSPLINE_STRUCTS_STD_H
#define BSPLINE_STRUCTS_STD_H
#include <stdlib.h>

///////////////////////////
// Single precision real //
///////////////////////////
struct UBspline_3d_s
{
  spline_code spcode;
  type_code tcode;
  float* restrict coefs;
  int x_stride, y_stride;
  Ugrid x_grid, y_grid, z_grid;
  BCtype_s xBC, yBC, zBC;
  size_t coefs_size;
};

///////////////////////////
// Double precision real //
///////////////////////////
struct UBspline_3d_d
{
  spline_code spcode;
  type_code tcode;
  double* restrict coefs;
  int x_stride, y_stride;
  Ugrid x_grid, y_grid, z_grid;
  BCtype_d xBC, yBC, zBC;
  size_t coefs_size;
};

//////////////////////////////
// Single precision complex //
//////////////////////////////
struct UBspline_3d_c
{
  spline_code spcode;
  type_code tcode;
  complex_float* restrict coefs;
  int x_stride, y_stride;
  Ugrid x_grid, y_grid, z_grid;
  BCtype_c xBC, yBC, zBC;
};

//////////////////////////////
// Double precision complex //
//////////////////////////////
struct UBspline_3d_z
{
  spline_code spcode;
  type_code tcode;
  complex_double* restrict coefs;
  int x_stride, y_stride;
  Ugrid x_grid, y_grid, z_grid;
  BCtype_z xBC, yBC, zBC;
};

#endif
