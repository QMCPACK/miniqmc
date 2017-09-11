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

#include "bspline_create.h"
#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 600
#endif
#ifndef __USE_XOPEN2K
#define __USE_XOPEN2K
#endif
#include <stdio.h>
#include <stdlib.h>

int posix_memalign(void **memptr, size_t alignment, size_t size);

////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
////       Helper functions for spline creation         ////
////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////

void find_coefs_1d_d(Ugrid grid, BCtype_d bc, double *data, intptr_t dstride,
                     double *coefs, intptr_t cstride);

void solve_deriv_interp_1d_s(float bands[], float coefs[], int M, int cstride)
{
  // Solve interpolating equations
  // First and last rows are different
  bands[4 * (0) + 1] /= bands[4 * (0) + 0];
  bands[4 * (0) + 2] /= bands[4 * (0) + 0];
  bands[4 * (0) + 3] /= bands[4 * (0) + 0];
  bands[4 * (0) + 0] = 1.0;
  bands[4 * (1) + 1] -= bands[4 * (1) + 0] * bands[4 * (0) + 1];
  bands[4 * (1) + 2] -= bands[4 * (1) + 0] * bands[4 * (0) + 2];
  bands[4 * (1) + 3] -= bands[4 * (1) + 0] * bands[4 * (0) + 3];
  bands[4 * (0) + 0] = 0.0;
  bands[4 * (1) + 2] /= bands[4 * (1) + 1];
  bands[4 * (1) + 3] /= bands[4 * (1) + 1];
  bands[4 * (1) + 1] = 1.0;

  // Now do rows 2 through M+1
  for (int row = 2; row < (M + 1); row++)
  {
    bands[4 * (row) + 1] -= bands[4 * (row) + 0] * bands[4 * (row - 1) + 2];
    bands[4 * (row) + 3] -= bands[4 * (row) + 0] * bands[4 * (row - 1) + 3];
    bands[4 * (row) + 2] /= bands[4 * (row) + 1];
    bands[4 * (row) + 3] /= bands[4 * (row) + 1];
    bands[4 * (row) + 0] = 0.0;
    bands[4 * (row) + 1] = 1.0;
  }

  // Do last row
  bands[4 * (M + 1) + 1] -= bands[4 * (M + 1) + 0] * bands[4 * (M - 1) + 2];
  bands[4 * (M + 1) + 3] -= bands[4 * (M + 1) + 0] * bands[4 * (M - 1) + 3];
  bands[4 * (M + 1) + 2] -= bands[4 * (M + 1) + 1] * bands[4 * (M) + 2];
  bands[4 * (M + 1) + 3] -= bands[4 * (M + 1) + 1] * bands[4 * (M) + 3];
  bands[4 * (M + 1) + 3] /= bands[4 * (M + 1) + 2];
  bands[4 * (M + 1) + 2] = 1.0;

  coefs[(M + 1) * cstride] = bands[4 * (M + 1) + 3];
  // Now back substitute up
  for (int row           = M; row > 0; row--)
    coefs[row * cstride] = bands[4 * (row) + 3] -
                           bands[4 * (row) + 2] * coefs[cstride * (row + 1)];

  // Finish with first row
  coefs[0] = bands[4 * (0) + 3] - bands[4 * (0) + 1] * coefs[1 * cstride] -
             bands[4 * (0) + 2] * coefs[2 * cstride];
}

// On input, bands should be filled with:
// row 0   :  abcdInitial from boundary conditions
// rows 1:M:  basis functions in first 3 cols, data in last
// row M+1 :  abcdFinal   from boundary conditions
// cstride gives the stride between values in coefs.
// On exit, coefs with contain interpolating B-spline coefs
void solve_periodic_interp_1d_s(float bands[], float coefs[], int M,
                                size_t cstride) // int M, int cstride)
{
  float lastCol[M];
  // Now solve:
  // First and last rows are different
  bands[4 * (0) + 2] /= bands[4 * (0) + 1];
  bands[4 * (0) + 0] /= bands[4 * (0) + 1];
  bands[4 * (0) + 3] /= bands[4 * (0) + 1];
  bands[4 * (0) + 1] = 1.0;
  bands[4 * (M - 1) + 1] -= bands[4 * (M - 1) + 2] * bands[4 * (0) + 0];
  bands[4 * (M - 1) + 3] -= bands[4 * (M - 1) + 2] * bands[4 * (0) + 3];
  bands[4 * (M - 1) + 2] = -bands[4 * (M - 1) + 2] * bands[4 * (0) + 2];
  lastCol[0]             = bands[4 * (0) + 0];

  for (int row = 1; row < (M - 1); row++)
  {
    bands[4 * (row) + 1] -= bands[4 * (row) + 0] * bands[4 * (row - 1) + 2];
    bands[4 * (row) + 3] -= bands[4 * (row) + 0] * bands[4 * (row - 1) + 3];
    lastCol[row]         = -bands[4 * (row) + 0] * lastCol[row - 1];
    bands[4 * (row) + 0] = 0.0;
    bands[4 * (row) + 2] /= bands[4 * (row) + 1];
    bands[4 * (row) + 3] /= bands[4 * (row) + 1];
    lastCol[row] /= bands[4 * (row) + 1];
    bands[4 * (row) + 1] = 1.0;
    if (row < (M - 2))
    {
      bands[4 * (M - 1) + 3] -= bands[4 * (M - 1) + 2] * bands[4 * (row) + 3];
      bands[4 * (M - 1) + 1] -= bands[4 * (M - 1) + 2] * lastCol[row];
      bands[4 * (M - 1) + 2] = -bands[4 * (M - 1) + 2] * bands[4 * (row) + 2];
    }
  }

  // Now do last row
  // The [2] element and [0] element are now on top of each other
  bands[4 * (M - 1) + 0] += bands[4 * (M - 1) + 2];
  bands[4 * (M - 1) + 1] -=
      bands[4 * (M - 1) + 0] * (bands[4 * (M - 2) + 2] + lastCol[M - 2]);
  bands[4 * (M - 1) + 3] -= bands[4 * (M - 1) + 0] * bands[4 * (M - 2) + 3];
  bands[4 * (M - 1) + 3] /= bands[4 * (M - 1) + 1];
  coefs[M * cstride] = bands[4 * (M - 1) + 3];
  for (int row = M - 2; row >= 0; row--)
    coefs[(row + 1) * cstride] =
        bands[4 * (row) + 3] -
        bands[4 * (row) + 2] * coefs[(row + 2) * cstride] -
        lastCol[row] * coefs[M * cstride];

  coefs[0 * cstride]       = coefs[M * cstride];
  coefs[(M + 1) * cstride] = coefs[1 * cstride];
  coefs[(M + 2) * cstride] = coefs[2 * cstride];
}

// On input, bands should be filled with:
// row 0   :  abcdInitial from boundary conditions
// rows 1:M:  basis functions in first 3 cols, data in last
// row M+1 :  abcdFinal   from boundary conditions
// cstride gives the stride between values in coefs.
// On exit, coefs with contain interpolating B-spline coefs
void solve_antiperiodic_interp_1d_s(float bands[], float coefs[], int M,
                                    int cstride)
{
  bands[4 * 0 + 0] *= -1.0;
  bands[4 * (M - 1) + 2] *= -1.0;

  float lastCol[M];
  // Now solve:
  // First and last rows are different
  bands[4 * (0) + 2] /= bands[4 * (0) + 1];
  bands[4 * (0) + 0] /= bands[4 * (0) + 1];
  bands[4 * (0) + 3] /= bands[4 * (0) + 1];
  bands[4 * (0) + 1] = 1.0;
  bands[4 * (M - 1) + 1] -= bands[4 * (M - 1) + 2] * bands[4 * (0) + 0];
  bands[4 * (M - 1) + 3] -= bands[4 * (M - 1) + 2] * bands[4 * (0) + 3];
  bands[4 * (M - 1) + 2] = -bands[4 * (M - 1) + 2] * bands[4 * (0) + 2];
  lastCol[0]             = bands[4 * (0) + 0];

  for (int row = 1; row < (M - 1); row++)
  {
    bands[4 * (row) + 1] -= bands[4 * (row) + 0] * bands[4 * (row - 1) + 2];
    bands[4 * (row) + 3] -= bands[4 * (row) + 0] * bands[4 * (row - 1) + 3];
    lastCol[row]         = -bands[4 * (row) + 0] * lastCol[row - 1];
    bands[4 * (row) + 0] = 0.0;
    bands[4 * (row) + 2] /= bands[4 * (row) + 1];
    bands[4 * (row) + 3] /= bands[4 * (row) + 1];
    lastCol[row] /= bands[4 * (row) + 1];
    bands[4 * (row) + 1] = 1.0;
    if (row < (M - 2))
    {
      bands[4 * (M - 1) + 3] -= bands[4 * (M - 1) + 2] * bands[4 * (row) + 3];
      bands[4 * (M - 1) + 1] -= bands[4 * (M - 1) + 2] * lastCol[row];
      bands[4 * (M - 1) + 2] = -bands[4 * (M - 1) + 2] * bands[4 * (row) + 2];
    }
  }

  // Now do last row
  // The [2] element and [0] element are now on top of each other
  bands[4 * (M - 1) + 0] += bands[4 * (M - 1) + 2];
  bands[4 * (M - 1) + 1] -=
      bands[4 * (M - 1) + 0] * (bands[4 * (M - 2) + 2] + lastCol[M - 2]);
  bands[4 * (M - 1) + 3] -= bands[4 * (M - 1) + 0] * bands[4 * (M - 2) + 3];
  bands[4 * (M - 1) + 3] /= bands[4 * (M - 1) + 1];
  coefs[M * cstride] = bands[4 * (M - 1) + 3];
  for (int row = M - 2; row >= 0; row--)
    coefs[(row + 1) * cstride] =
        bands[4 * (row) + 3] -
        bands[4 * (row) + 2] * coefs[(row + 2) * cstride] -
        lastCol[row] * coefs[M * cstride];

  coefs[0 * cstride]       = -coefs[M * cstride];
  coefs[(M + 1) * cstride] = -coefs[1 * cstride];
  coefs[(M + 2) * cstride] = -coefs[2 * cstride];
}

#ifdef HIGH_PRECISION
void find_coefs_1d_s(Ugrid grid, BCtype_s bc, float *data, intptr_t dstride,
                     float *coefs, intptr_t cstride)
{
  BCtype_d d_bc;
  double *d_data, *d_coefs;

  d_bc.lCode = bc.lCode;
  d_bc.rCode = bc.rCode;
  d_bc.lVal  = bc.lVal;
  d_bc.rVal  = bc.rVal;
  int M      = grid.num, N;
  if (bc.lCode == PERIODIC || bc.lCode == ANTIPERIODIC)
    N = M + 3;
  else
    N = M + 2;

  d_data  = malloc(N * sizeof(double));
  d_coefs = malloc(N * sizeof(double));
  for (int i  = 0; i < M; i++)
    d_data[i] = data[i * dstride];
  find_coefs_1d_d(grid, d_bc, d_data, 1, d_coefs, 1);
  for (int i           = 0; i < N; i++)
    coefs[i * cstride] = d_coefs[i];
  free(d_data);
  free(d_coefs);
}

#else
void find_coefs_1d_s(Ugrid grid, BCtype_s bc, float *data, intptr_t dstride,
                     float *coefs, intptr_t cstride)
{
  size_t M       = grid.num;
  float basis[4] = {1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0, 0.0};
  if (bc.lCode == PERIODIC || bc.lCode == ANTIPERIODIC)
  {
#ifdef HAVE_C_VARARRAYS
    float bands[4 * M];
#else
    float *bands = malloc(4 * M * sizeof(float));
#endif
    for (size_t i = 0; i < M; i++)
    {
      bands[4 * i + 0] = basis[0];
      bands[4 * i + 1] = basis[1];
      bands[4 * i + 2] = basis[2];
      bands[4 * i + 3] = data[i * dstride];
    }
    if (bc.lCode == PERIODIC)
      solve_periodic_interp_1d_s(bands, coefs, M, cstride);
    else
      solve_antiperiodic_interp_1d_s(bands, coefs, M, cstride);
#ifndef HAVE_C_VARARRAYS
    free(bands);
#endif
  }
  else
  {
    // Setup boundary conditions
    float abcd_left[4], abcd_right[4];
    // Left boundary
    if (bc.lCode == FLAT || bc.lCode == NATURAL) bc.lVal = 0.0;
    if (bc.lCode == FLAT || bc.lCode == DERIV1)
    {
      abcd_left[0] = -0.5 * grid.delta_inv;
      abcd_left[1] = 0.0 * grid.delta_inv;
      abcd_left[2] = 0.5 * grid.delta_inv;
      abcd_left[3] = bc.lVal;
    }
    if (bc.lCode == NATURAL || bc.lCode == DERIV2)
    {
      abcd_left[0] = 1.0 * grid.delta_inv * grid.delta_inv;
      abcd_left[1] = -2.0 * grid.delta_inv * grid.delta_inv;
      abcd_left[2] = 1.0 * grid.delta_inv * grid.delta_inv;
      abcd_left[3] = bc.lVal;
    }

    // Right boundary
    if (bc.rCode == FLAT || bc.rCode == NATURAL) bc.rVal = 0.0;
    if (bc.rCode == FLAT || bc.rCode == DERIV1)
    {
      abcd_right[0] = -0.5 * grid.delta_inv;
      abcd_right[1] = 0.0 * grid.delta_inv;
      abcd_right[2] = 0.5 * grid.delta_inv;
      abcd_right[3] = bc.rVal;
    }
    if (bc.rCode == NATURAL || bc.rCode == DERIV2)
    {
      abcd_right[0] = 1.0 * grid.delta_inv * grid.delta_inv;
      abcd_right[1] = -2.0 * grid.delta_inv * grid.delta_inv;
      abcd_right[2] = 1.0 * grid.delta_inv * grid.delta_inv;
      abcd_right[3] = bc.rVal;
    }
#ifdef HAVE_C_VARARRAYS
    float bands[4 * (M + 2)];
#else
    float *bands = malloc((M + 2) * 4 * sizeof(float));
#endif
    for (int i = 0; i < 4; i++)
    {
      bands[4 * (0) + i]     = abcd_left[i];
      bands[4 * (M + 1) + i] = abcd_right[i];
    }
    for (int i = 0; i < M; i++)
    {
      for (int j               = 0; j < 3; j++)
        bands[4 * (i + 1) + j] = basis[j];

      bands[4 * (i + 1) + 3] = data[i * dstride];
    }
    // Now, solve for coefficients
    solve_deriv_interp_1d_s(bands, coefs, M, cstride);
#ifndef HAVE_C_VARARRAYS
    free(bands);
#endif
  }
}

#endif

// On input, bands should be filled with:
// row 0   :  abcdInitial from boundary conditions
// rows 1:M:  basis functions in first 3 cols, data in last
// row M+1 :  abcdFinal   from boundary conditions
// cstride gives the stride between values in coefs.
// On exit, coefs with contain interpolating B-spline coefs
void solve_deriv_interp_1d_d(double bands[], double coefs[], int M, int cstride)
{
  // Solve interpolating equations
  // First and last rows are different
  bands[4 * (0) + 1] /= bands[4 * (0) + 0];
  bands[4 * (0) + 2] /= bands[4 * (0) + 0];
  bands[4 * (0) + 3] /= bands[4 * (0) + 0];
  bands[4 * (0) + 0] = 1.0;
  bands[4 * (1) + 1] -= bands[4 * (1) + 0] * bands[4 * (0) + 1];
  bands[4 * (1) + 2] -= bands[4 * (1) + 0] * bands[4 * (0) + 2];
  bands[4 * (1) + 3] -= bands[4 * (1) + 0] * bands[4 * (0) + 3];
  bands[4 * (0) + 0] = 0.0;
  bands[4 * (1) + 2] /= bands[4 * (1) + 1];
  bands[4 * (1) + 3] /= bands[4 * (1) + 1];
  bands[4 * (1) + 1] = 1.0;

  // Now do rows 2 through M+1
  for (int row = 2; row < (M + 1); row++)
  {
    bands[4 * (row) + 1] -= bands[4 * (row) + 0] * bands[4 * (row - 1) + 2];
    bands[4 * (row) + 3] -= bands[4 * (row) + 0] * bands[4 * (row - 1) + 3];
    bands[4 * (row) + 2] /= bands[4 * (row) + 1];
    bands[4 * (row) + 3] /= bands[4 * (row) + 1];
    bands[4 * (row) + 0] = 0.0;
    bands[4 * (row) + 1] = 1.0;
  }

  // Do last row
  bands[4 * (M + 1) + 1] -= bands[4 * (M + 1) + 0] * bands[4 * (M - 1) + 2];
  bands[4 * (M + 1) + 3] -= bands[4 * (M + 1) + 0] * bands[4 * (M - 1) + 3];
  bands[4 * (M + 1) + 2] -= bands[4 * (M + 1) + 1] * bands[4 * (M) + 2];
  bands[4 * (M + 1) + 3] -= bands[4 * (M + 1) + 1] * bands[4 * (M) + 3];
  bands[4 * (M + 1) + 3] /= bands[4 * (M + 1) + 2];
  bands[4 * (M + 1) + 2] = 1.0;

  coefs[(M + 1) * cstride] = bands[4 * (M + 1) + 3];
  // Now back substitute up
  for (int row = M; row > 0; row--)
  {
    coefs[row * cstride] = bands[4 * (row) + 3] -
                           bands[4 * (row) + 2] * coefs[cstride * (row + 1)];
  }

  // Finish with first row
  coefs[0] = bands[4 * (0) + 3] - bands[4 * (0) + 1] * coefs[1 * cstride] -
             bands[4 * (0) + 2] * coefs[2 * cstride];
}

// On input, bands should be filled with:
// row 0   :  abcdInitial from boundary conditions
// rows 1:M:  basis functions in first 3 cols, data in last
// row M+1 :  abcdFinal   from boundary conditions
// cstride gives the stride between values in coefs.
// On exit, coefs with contain interpolating B-spline coefs
void solve_periodic_interp_1d_d(double bands[], double coefs[], int M,
                                intptr_t cstride)
{
  double lastCol[M];
  // Now solve:
  // First and last rows are different
  bands[4 * (0) + 2] /= bands[4 * (0) + 1];
  bands[4 * (0) + 0] /= bands[4 * (0) + 1];
  bands[4 * (0) + 3] /= bands[4 * (0) + 1];
  bands[4 * (0) + 1] = 1.0;
  bands[4 * (M - 1) + 1] -= bands[4 * (M - 1) + 2] * bands[4 * (0) + 0];
  bands[4 * (M - 1) + 3] -= bands[4 * (M - 1) + 2] * bands[4 * (0) + 3];
  bands[4 * (M - 1) + 2] = -bands[4 * (M - 1) + 2] * bands[4 * (0) + 2];
  lastCol[0]             = bands[4 * (0) + 0];

  for (int row = 1; row < (M - 1); row++)
  {
    bands[4 * (row) + 1] -= bands[4 * (row) + 0] * bands[4 * (row - 1) + 2];
    bands[4 * (row) + 3] -= bands[4 * (row) + 0] * bands[4 * (row - 1) + 3];
    lastCol[row]         = -bands[4 * (row) + 0] * lastCol[row - 1];
    bands[4 * (row) + 0] = 0.0;
    bands[4 * (row) + 2] /= bands[4 * (row) + 1];
    bands[4 * (row) + 3] /= bands[4 * (row) + 1];
    lastCol[row] /= bands[4 * (row) + 1];
    bands[4 * (row) + 1] = 1.0;
    if (row < (M - 2))
    {
      bands[4 * (M - 1) + 3] -= bands[4 * (M - 1) + 2] * bands[4 * (row) + 3];
      bands[4 * (M - 1) + 1] -= bands[4 * (M - 1) + 2] * lastCol[row];
      bands[4 * (M - 1) + 2] = -bands[4 * (M - 1) + 2] * bands[4 * (row) + 2];
    }
  }

  // Now do last row
  // The [2] element and [0] element are now on top of each other
  bands[4 * (M - 1) + 0] += bands[4 * (M - 1) + 2];
  bands[4 * (M - 1) + 1] -=
      bands[4 * (M - 1) + 0] * (bands[4 * (M - 2) + 2] + lastCol[M - 2]);
  bands[4 * (M - 1) + 3] -= bands[4 * (M - 1) + 0] * bands[4 * (M - 2) + 3];
  bands[4 * (M - 1) + 3] /= bands[4 * (M - 1) + 1];
  coefs[M * cstride] = bands[4 * (M - 1) + 3];
  for (int row = M - 2; row >= 0; row--)
    coefs[(row + 1) * cstride] =
        bands[4 * (row) + 3] -
        bands[4 * (row) + 2] * coefs[(row + 2) * cstride] -
        lastCol[row] * coefs[M * cstride];

  coefs[0 * cstride]       = coefs[M * cstride];
  coefs[(M + 1) * cstride] = coefs[1 * cstride];
  coefs[(M + 2) * cstride] = coefs[2 * cstride];
}

// On input, bands should be filled with:
// row 0   :  abcdInitial from boundary conditions
// rows 1:M:  basis functions in first 3 cols, data in last
// row M+1 :  abcdFinal   from boundary conditions
// cstride gives the stride between values in coefs.
// On exit, coefs with contain interpolating B-spline coefs
void solve_antiperiodic_interp_1d_d(double bands[], double coefs[], int M,
                                    int cstride)
{
  double lastCol[M];
  bands[4 * 0 + 0] *= -1.0;
  bands[4 * (M - 1) + 2] *= -1.0;
  // Now solve:
  // First and last rows are different
  bands[4 * (0) + 2] /= bands[4 * (0) + 1];
  bands[4 * (0) + 0] /= bands[4 * (0) + 1];
  bands[4 * (0) + 3] /= bands[4 * (0) + 1];
  bands[4 * (0) + 1] = 1.0;
  bands[4 * (M - 1) + 1] -= bands[4 * (M - 1) + 2] * bands[4 * (0) + 0];
  bands[4 * (M - 1) + 3] -= bands[4 * (M - 1) + 2] * bands[4 * (0) + 3];
  bands[4 * (M - 1) + 2] = -bands[4 * (M - 1) + 2] * bands[4 * (0) + 2];
  lastCol[0]             = bands[4 * (0) + 0];

  for (int row = 1; row < (M - 1); row++)
  {
    bands[4 * (row) + 1] -= bands[4 * (row) + 0] * bands[4 * (row - 1) + 2];
    bands[4 * (row) + 3] -= bands[4 * (row) + 0] * bands[4 * (row - 1) + 3];
    lastCol[row]         = -bands[4 * (row) + 0] * lastCol[row - 1];
    bands[4 * (row) + 0] = 0.0;
    bands[4 * (row) + 2] /= bands[4 * (row) + 1];
    bands[4 * (row) + 3] /= bands[4 * (row) + 1];
    lastCol[row] /= bands[4 * (row) + 1];
    bands[4 * (row) + 1] = 1.0;
    if (row < (M - 2))
    {
      bands[4 * (M - 1) + 3] -= bands[4 * (M - 1) + 2] * bands[4 * (row) + 3];
      bands[4 * (M - 1) + 1] -= bands[4 * (M - 1) + 2] * lastCol[row];
      bands[4 * (M - 1) + 2] = -bands[4 * (M - 1) + 2] * bands[4 * (row) + 2];
    }
  }

  // Now do last row
  // The [2] element and [0] element are now on top of each other
  bands[4 * (M - 1) + 0] += bands[4 * (M - 1) + 2];
  bands[4 * (M - 1) + 1] -=
      bands[4 * (M - 1) + 0] * (bands[4 * (M - 2) + 2] + lastCol[M - 2]);
  bands[4 * (M - 1) + 3] -= bands[4 * (M - 1) + 0] * bands[4 * (M - 2) + 3];
  bands[4 * (M - 1) + 3] /= bands[4 * (M - 1) + 1];
  coefs[M * cstride] = bands[4 * (M - 1) + 3];
  for (int row = M - 2; row >= 0; row--)
    coefs[(row + 1) * cstride] =
        bands[4 * (row) + 3] -
        bands[4 * (row) + 2] * coefs[(row + 2) * cstride] -
        lastCol[row] * coefs[M * cstride];

  coefs[0 * cstride]       = -coefs[M * cstride];
  coefs[(M + 1) * cstride] = -coefs[1 * cstride];
  coefs[(M + 2) * cstride] = -coefs[2 * cstride];
}

void find_coefs_1d_d(Ugrid grid, BCtype_d bc, double *data, intptr_t dstride,
                     double *coefs, intptr_t cstride)
{
  int M           = grid.num;
  double basis[4] = {1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0, 0.0};
  if (bc.lCode == PERIODIC || bc.lCode == ANTIPERIODIC)
  {
#ifdef HAVE_C_VARARRAYS
    double bands[M * 4];
#else
    double *bands = malloc(4 * M * sizeof(double));
#endif
    for (int i = 0; i < M; i++)
    {
      bands[4 * i + 0] = basis[0];
      bands[4 * i + 1] = basis[1];
      bands[4 * i + 2] = basis[2];
      bands[4 * i + 3] = data[i * dstride];
    }
    if (bc.lCode == ANTIPERIODIC)
      solve_antiperiodic_interp_1d_d(bands, coefs, M, cstride);
    else
      solve_periodic_interp_1d_d(bands, coefs, M, cstride);

#ifndef HAVE_C_VARARRAYS
    free(bands);
#endif
  }
  else
  {
    // Setup boundary conditions
    double abcd_left[4], abcd_right[4];
    // Left boundary
    if (bc.lCode == FLAT || bc.lCode == NATURAL) bc.lVal = 0.0;
    if (bc.lCode == FLAT || bc.lCode == DERIV1)
    {
      abcd_left[0] = -0.5 * grid.delta_inv;
      abcd_left[1] = 0.0 * grid.delta_inv;
      abcd_left[2] = 0.5 * grid.delta_inv;
      abcd_left[3] = bc.lVal;
    }
    if (bc.lCode == NATURAL || bc.lCode == DERIV2)
    {
      abcd_left[0] = 1.0 * grid.delta_inv * grid.delta_inv;
      abcd_left[1] = -2.0 * grid.delta_inv * grid.delta_inv;
      abcd_left[2] = 1.0 * grid.delta_inv * grid.delta_inv;
      abcd_left[3] = bc.lVal;
    }

    // Right boundary
    if (bc.rCode == FLAT || bc.rCode == NATURAL) bc.rVal = 0.0;
    if (bc.rCode == FLAT || bc.rCode == DERIV1)
    {
      abcd_right[0] = -0.5 * grid.delta_inv;
      abcd_right[1] = 0.0 * grid.delta_inv;
      abcd_right[2] = 0.5 * grid.delta_inv;
      abcd_right[3] = bc.rVal;
    }
    if (bc.rCode == NATURAL || bc.rCode == DERIV2)
    {
      abcd_right[0] = 1.0 * grid.delta_inv * grid.delta_inv;
      abcd_right[1] = -2.0 * grid.delta_inv * grid.delta_inv;
      abcd_right[2] = 1.0 * grid.delta_inv * grid.delta_inv;
      abcd_right[3] = bc.rVal;
    }
#ifdef HAVE_C_VARARRAYS
    double bands[(M + 2) * 4];
#else
    double *bands = malloc((M + 2) * 4 * sizeof(double));
#endif
    for (int i = 0; i < 4; i++)
    {
      bands[4 * (0) + i]     = abcd_left[i];
      bands[4 * (M + 1) + i] = abcd_right[i];
    }
    for (int i = 0; i < M; i++)
    {
      for (int j               = 0; j < 3; j++)
        bands[4 * (i + 1) + j] = basis[j];

      bands[4 * (i + 1) + 3] = data[i * dstride];
    }
    // Now, solve for coefficients
    solve_deriv_interp_1d_d(bands, coefs, M, cstride);
#ifndef HAVE_C_VARARRAYS
    free(bands);
#endif
  }
}

void destroy_UBspline(Bspline *spline)
{
  free(spline->coefs);
  free(spline);
}

void destroy_multi_UBspline(Bspline *spline);

void destroy_Bspline(void *spline)
{
  Bspline *sp = (Bspline *)spline;
  if (sp->sp_code <= U3D)
    destroy_UBspline(sp);
  else if (sp->sp_code <= MULTI_U3D)
    destroy_multi_UBspline(sp);
  else
    fprintf(stderr, "Error in destroy_Bspline:  invalide spline code %d.\n",
            sp->sp_code);
}
