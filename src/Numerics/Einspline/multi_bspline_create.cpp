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

#include "multi_bspline_create.h"
#include "bspline_create.h"
#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 600
#endif
#ifndef __USE_XOPEN2K
#define __USE_XOPEN2K
#endif
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

int posix_memalign(void **memptr, size_t alignment, size_t size);

////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
////       Helper functions for spline creation         ////
////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////

void set_multi_UBspline_3d_s(multi_UBspline_3d_s *spline, int num, float *data)
{
  int Mx = spline->x_grid.num;
  int My = spline->y_grid.num;
  int Mz = spline->z_grid.num;
  int Nx, Ny, Nz;

  if (spline->xBC.lCode == PERIODIC || spline->xBC.lCode == ANTIPERIODIC)
    Nx = Mx + 3;
  else
    Nx = Mx + 2;
  if (spline->yBC.lCode == PERIODIC || spline->yBC.lCode == ANTIPERIODIC)
    Ny = My + 3;
  else
    Ny = My + 2;
  if (spline->zBC.lCode == PERIODIC || spline->zBC.lCode == ANTIPERIODIC)
    Nz = Mz + 3;
  else
    Nz = Mz + 2;

  float *coefs = spline->coefs + num;

  intptr_t zs = spline->z_stride;
// First, solve in the X-direction
#pragma omp parallel for
  for (int iy = 0; iy < My; iy++)
    for (int iz = 0; iz < Mz; iz++)
    {
      intptr_t doffset = iy * Mz + iz;
      intptr_t coffset = (iy * Nz + iz) * zs;
      find_coefs_1d_s(spline->x_grid, spline->xBC, data + doffset,
                      (intptr_t)(My * Mz), coefs + coffset,
                      (intptr_t)(Ny * Nz) * zs);
    }

// Now, solve in the Y-direction
#pragma omp parallel for
  for (int ix = 0; ix < Nx; ix++)
    for (int iz = 0; iz < Nz; iz++)
    {
      intptr_t doffset = (ix * Ny * Nz + iz) * zs;
      intptr_t coffset = (ix * Ny * Nz + iz) * zs;
      find_coefs_1d_s(spline->y_grid, spline->yBC, coefs + doffset,
                      (intptr_t)Nz * zs, coefs + coffset, (intptr_t)Nz * zs);
    }

// Now, solve in the Z-direction
#pragma omp parallel for
  for (int ix = 0; ix < Nx; ix++)
    for (int iy = 0; iy < Ny; iy++)
    {
      intptr_t doffset = ((ix * Ny + iy) * Nz) * zs;
      intptr_t coffset = ((ix * Ny + iy) * Nz) * zs;
      find_coefs_1d_s(spline->z_grid, spline->zBC, coefs + doffset, zs,
                      coefs + coffset, zs);
    }
}

void set_multi_UBspline_3d_s_d(multi_UBspline_3d_s *spline, int num,
                               double *data)
{

  BCtype_d xBC, yBC, zBC;
  xBC.lCode = spline->xBC.lCode;
  xBC.rCode = spline->xBC.rCode;
  yBC.lCode = spline->yBC.lCode;
  yBC.rCode = spline->yBC.rCode;
  zBC.lCode = spline->zBC.lCode;
  zBC.rCode = spline->zBC.rCode;
  xBC.lVal  = spline->xBC.lVal;
  xBC.rVal  = spline->xBC.rVal;
  yBC.lVal  = spline->yBC.lVal;
  yBC.rVal  = spline->yBC.rVal;
  zBC.lVal  = spline->zBC.lVal;
  zBC.rVal  = spline->zBC.rVal;

  int Mx = spline->x_grid.num;
  int My = spline->y_grid.num;
  int Mz = spline->z_grid.num;
  int Nx, Ny, Nz;

  if (spline->xBC.lCode == PERIODIC || spline->xBC.lCode == ANTIPERIODIC)
    Nx = Mx + 3;
  else
    Nx = Mx + 2;
  if (spline->yBC.lCode == PERIODIC || spline->yBC.lCode == ANTIPERIODIC)
    Ny = My + 3;
  else
    Ny = My + 2;
  if (spline->zBC.lCode == PERIODIC || spline->zBC.lCode == ANTIPERIODIC)
    Nz = Mz + 3;
  else
    Nz = Mz + 2;

  double *spline_tmp = (double *)malloc(sizeof(double) * Nx * Ny * Nz);

// First, solve in the X-direction
#pragma omp parallel for
  for (int iy = 0; iy < My; iy++)
    for (int iz = 0; iz < Mz; iz++)
    {
      intptr_t doffset = iy * Mz + iz;
      intptr_t coffset = iy * Nz + iz;
      find_coefs_1d_d(spline->x_grid, xBC, data + doffset, My * Mz,
                      spline_tmp + coffset, Ny * Nz);
    }

// Now, solve in the Y-direction
#pragma omp parallel for
  for (int ix = 0; ix < Nx; ix++)
    for (int iz = 0; iz < Nz; iz++)
    {
      intptr_t doffset = ix * Ny * Nz + iz;
      intptr_t coffset = ix * Ny * Nz + iz;
      find_coefs_1d_d(spline->y_grid, yBC, spline_tmp + doffset, Nz,
                      spline_tmp + coffset, Nz);
    }

// Now, solve in the Z-direction
#pragma omp parallel for
  for (int ix = 0; ix < Nx; ix++)
    for (int iy = 0; iy < Ny; iy++)
    {
      intptr_t doffset = (ix * Ny + iy) * Nz;
      intptr_t coffset = (ix * Ny + iy) * Nz;
      find_coefs_1d_d(spline->z_grid, zBC, spline_tmp + doffset, 1,
                      spline_tmp + coffset, 1);
    }

  {
//    const double* restrict i_ptr=spline_tmp;
#pragma omp parallel for
    for (int ix = 0; ix < Nx; ++ix)
    {
      const double *restrict i_ptr = spline_tmp + ix * Ny * Nz;
      for (int iy = 0; iy < Ny; ++iy)
        for (int iz = 0; iz < Nz; ++iz)
          spline->coefs[ix * spline->x_stride + iy * spline->y_stride +
                        iz * spline->z_stride + num] = (float)(*i_ptr++);
    }
  }

  free(spline_tmp);
}

void set_multi_UBspline_3d_d(multi_UBspline_3d_d *spline, int num, double *data)
{
  int Mx = spline->x_grid.num;
  int My = spline->y_grid.num;
  int Mz = spline->z_grid.num;
  int Nx, Ny, Nz;

  if (spline->xBC.lCode == PERIODIC || spline->xBC.lCode == ANTIPERIODIC)
    Nx = Mx + 3;
  else
    Nx = Mx + 2;
  if (spline->yBC.lCode == PERIODIC || spline->yBC.lCode == ANTIPERIODIC)
    Ny = My + 3;
  else
    Ny = My + 2;
  if (spline->zBC.lCode == PERIODIC || spline->zBC.lCode == ANTIPERIODIC)
    Nz = Mz + 3;
  else
    Nz = Mz + 2;

  double *coefs = spline->coefs + num;
  intptr_t zs   = spline->z_stride;

// First, solve in the X-direction
#pragma omp parallel for
  for (int iy = 0; iy < My; iy++)
    for (int iz = 0; iz < Mz; iz++)
    {
      intptr_t doffset = iy * Mz + iz;
      intptr_t coffset = (iy * Nz + iz) * zs;
      find_coefs_1d_d(spline->x_grid, spline->xBC, data + doffset,
                      (intptr_t)My * Mz, coefs + coffset,
                      (intptr_t)Ny * Nz * zs);
    }

// Now, solve in the Y-direction
#pragma omp parallel for
  for (int ix = 0; ix < Nx; ix++)
    for (int iz = 0; iz < Nz; iz++)
    {
      intptr_t doffset = (ix * Ny * Nz + iz) * zs;
      intptr_t coffset = (ix * Ny * Nz + iz) * zs;
      find_coefs_1d_d(spline->y_grid, spline->yBC, coefs + doffset,
                      (intptr_t)Nz * zs, coefs + coffset, (intptr_t)Nz * zs);
    }

// Now, solve in the Z-direction
#pragma omp parallel for
  for (int ix = 0; ix < Nx; ix++)
    for (int iy = 0; iy < Ny; iy++)
    {
      intptr_t doffset = (ix * Ny + iy) * Nz * zs;
      intptr_t coffset = (ix * Ny + iy) * Nz * zs;
      find_coefs_1d_d(spline->z_grid, spline->zBC, coefs + doffset,
                      (intptr_t)zs, coefs + coffset, (intptr_t)zs);
    }
}

void destroy_multi_UBspline(Bspline *spline)
{
  free(spline->coefs);
  free(spline);
}
