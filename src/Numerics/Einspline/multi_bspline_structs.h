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
  float *restrict coefs;
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
  float *restrict coefs;
  intptr_t x_stride, y_stride;
  Ugrid x_grid, y_grid;
  BCtype_s xBC, yBC;
  int num_splines;
} multi_UBspline_2d_s;

typedef struct
{
  spline_code spcode;
  type_code tcode;
  float *restrict coefs;
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
  double *restrict coefs;
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
  double *restrict coefs;
  intptr_t x_stride, y_stride;
  Ugrid x_grid, y_grid;
  BCtype_d xBC, yBC;
  int num_splines;
} multi_UBspline_2d_d;

class SplineCoefBase
{
public:
  SplineCoefBase(int Nx, int Ny, int Nz, int Num_splines) : nx(Nx), ny(Ny), nz(Nz), num_splines(Num_splines) {}

  virtual bool allocate_spline() = 0;
  virtual size_t get_coefs_size() = 0;
  virtual double *get_coefs(int ix, int iy, int iz) = 0;
  virtual void set_coeff(int ix, int iy, int iz, int spline_index, double val) = 0;
  virtual void free() {}
  int get_num_splines() { return num_splines; }

protected:
  int nx;
  int ny;
  int nz;
  int num_splines;
};

void *einspline_alloc(size_t N, size_t align);

class AlignedLocalSpline : public SplineCoefBase
{
public:
  AlignedLocalSpline(int Nx, int Ny, int Nz, int Num_splines): SplineCoefBase(Nx, Ny, Nz, Num_splines)
  {
  }
  bool allocate_spline() override
  {
     const int ND = QMC_CLINE / sizeof(double);
     int N =
        (num_splines % ND) ? (num_splines + ND - num_splines % ND) : num_splines;

      x_stride = (size_t)ny * (size_t)nz * (size_t)N;
      y_stride = nz * N;
      z_stride = N;

      coefs_size = (size_t)nx * x_stride;
      coefs =
        (double *)einspline_alloc(sizeof(double) * coefs_size, QMC_CLINE);

      if (coefs == NULL) return false;
      return true;

  }

  size_t get_coefs_size() override
  {
    return coefs_size;
  }

  double *get_coefs(int ix, int iy, int iz)  override
  {
    return coefs + (ix*x_stride + iy*y_stride + iz*z_stride);
  }

  void set_coeff(int ix, int iy, int iz, int spline_index, double val) override
  {
    coefs[iz*z_stride + iy*y_stride + iz*z_stride + spline_index] = val;
  }


private:
  size_t x_stride;
  size_t y_stride;
  size_t z_stride;
  size_t coefs_size;
  double *coefs;
};

typedef struct
{
  spline_code spcode;
  type_code tcode;

  // class to manage coefficients
  SplineCoefBase* coefbase;

  // only used for reference version
  double *restrict coefs;
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
  complex_float *restrict coefs;
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
  complex_float *restrict coefs;
  intptr_t x_stride, y_stride;
  Ugrid x_grid, y_grid;
  BCtype_c xBC, yBC;
  int num_splines;
  // temporary storage for laplacian components
  complex_float *restrict lapl2;
} multi_UBspline_2d_c;

typedef struct
{
  spline_code spcode;
  type_code tcode;
  complex_float *restrict coefs;
  intptr_t x_stride, y_stride, z_stride;
  Ugrid x_grid, y_grid, z_grid;
  BCtype_c xBC, yBC, zBC;
  int num_splines;
  size_t coefs_size;
  // temporary storage for laplacian components
  complex_float *restrict lapl3;
} multi_UBspline_3d_c;

//////////////////////////////
// Double precision complex //
//////////////////////////////
typedef struct
{
  spline_code spcode;
  type_code tcode;
  complex_double *restrict coefs;
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
  complex_double *restrict coefs;
  intptr_t x_stride, y_stride;
  Ugrid x_grid, y_grid;
  BCtype_z xBC, yBC;
  int num_splines;
  // temporary storage for laplacian components
  complex_double *restrict lapl2;
} multi_UBspline_2d_z;

typedef struct
{
  spline_code spcode;
  type_code tcode;
  complex_double *restrict coefs;
  intptr_t x_stride, y_stride, z_stride;
  Ugrid x_grid, y_grid, z_grid;
  BCtype_z xBC, yBC, zBC;
  int num_splines;
  size_t coefs_size;
  // temporary storage for laplacian components
  complex_double *restrict lapl3;
} multi_UBspline_3d_z;

#endif
