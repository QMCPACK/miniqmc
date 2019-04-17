/////////////////////////////////////////////////////////////////////////////
//  einspline:  a library for creating and evaluating B-splines            //
//  Modified 2018 Peter Doak
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
/** @file einspline_allocator.c
 *
 * Gather only 3d_d/3d_s allocation routines.
 * Original implementations are einspline/ *_create.c
 */
#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include "Devices.h"
#include "config.h"
#include "Numerics/Einspline/bspline.h"
#include "Numerics/Spline2/einspline_allocator.h"

#if defined(HAVE_POSIX_MEMALIGN)

int posix_memalign(void** memptr, size_t alignment, size_t size);

void* einspline_alloc(size_t size, size_t alignment)
{
  void* ptr;
  int ret = posix_memalign(&ptr, alignment, size);
  return ptr;
}

void einspline_free(void* ptr) { free(ptr); }

#else

void* einspline_alloc(size_t size, size_t alignment)
{
  size += (alignment - 1) + sizeof(void*);
  void* ptr = malloc(size);
  if (ptr == NULL)
    return NULL;
  else
  {
    void* shifted          = (char*)ptr + sizeof(void*); // make room to save original pointer
    size_t offset          = alignment - (size_t)shifted % (size_t)alignment;
    void* aligned          = (char*)shifted + offset;
    *((void**)aligned - 1) = ptr;
    return aligned;
  }
}

void einspline_free(void* aligned)
{
  void* ptr = *((void**)aligned - 1);
  free(ptr);
}
#endif

template<Devices D>
void einspline_create_multi_UBspline_3d_s(multi_UBspline_3d_s<D>*& restrict spline, Ugrid x_grid,
                                          Ugrid y_grid, Ugrid z_grid, BCtype_s xBC, BCtype_s yBC,
                                          BCtype_s zBC, int num_splines)
{
  // Create new spline
  spline = new multi_UBspline_3d_s<D>;
  if (!spline)
  {
    fprintf(stderr, "Out of memory allocating spline in create_multi_UBspline_3d_s.\n");
    abort();
  }
  spline->spcode      = MULTI_U3D;
  spline->tcode       = SINGLE_REAL;
  spline->xBC         = xBC;
  spline->yBC         = yBC;
  spline->zBC         = zBC;
  spline->num_splines = num_splines;
  // Setup internal variables
  int Mx = x_grid.num;
  int My = y_grid.num;
  int Mz = z_grid.num;
  int Nx, Ny, Nz;

  if (xBC.lCode == PERIODIC || xBC.lCode == ANTIPERIODIC)
    Nx = Mx + 3;
  else
    Nx = Mx + 2;
  x_grid.delta     = (x_grid.end - x_grid.start) / (double)(Nx - 3);
  x_grid.delta_inv = 1.0 / x_grid.delta;
  spline->x_grid   = x_grid;

  if (yBC.lCode == PERIODIC || yBC.lCode == ANTIPERIODIC)
    Ny = My + 3;
  else
    Ny = My + 2;
  y_grid.delta     = (y_grid.end - y_grid.start) / (double)(Ny - 3);
  y_grid.delta_inv = 1.0 / y_grid.delta;
  spline->y_grid   = y_grid;

  if (zBC.lCode == PERIODIC || zBC.lCode == ANTIPERIODIC)
    Nz = Mz + 3;
  else
    Nz = Mz + 2;
  z_grid.delta     = (z_grid.end - z_grid.start) / (double)(Nz - 3);
  z_grid.delta_inv = 1.0 / z_grid.delta;
  spline->z_grid   = z_grid;

  const int ND     = QMC_CLINE / sizeof(float);
  const size_t N   = ((num_splines + ND - 1) / ND) * ND;
  spline->x_stride = (size_t)Ny * (size_t)Nz * (size_t)N;
  spline->y_stride = (size_t)Nz * N;
  spline->z_stride = N;

  spline->coefs_size = (size_t)Nx * spline->x_stride;

  einspline_create_multi_UBspline_3d_s_coefs<D>(spline, Nx, Ny, Nz, N);

  if (!spline->coefs)
  {
    fprintf(stderr,
            "Out of memory allocating spline coefficients in "
            "create_multi_UBspline_3d_s.\n");
    abort();
  }
}


template<Devices D>
void einspline_create_multi_UBspline_3d_d(multi_UBspline_3d_d<D>*& spline, Ugrid x_grid,
                                          Ugrid y_grid, Ugrid z_grid, BCtype_d xBC, BCtype_d yBC,
                                          BCtype_d zBC, int num_splines)
{
  // Create new spline
  spline = new multi_UBspline_3d_d<D>;

  if (!spline)
  {
    fprintf(stderr, "Out of memory allocating spline in create_multi_UBspline_3d_d.\n");
    abort();
  }
  spline->spcode      = MULTI_U3D;
  spline->tcode       = DOUBLE_REAL;
  spline->xBC         = xBC;
  spline->yBC         = yBC;
  spline->zBC         = zBC;
  spline->num_splines = num_splines;

  // Setup internal variables
  int Mx = x_grid.num;
  int My = y_grid.num;
  int Mz = z_grid.num;
  int Nx, Ny, Nz;

  if (xBC.lCode == PERIODIC || xBC.lCode == ANTIPERIODIC)
    Nx = Mx + 3;
  else
    Nx = Mx + 2;
  x_grid.delta     = (x_grid.end - x_grid.start) / (double)(Nx - 3);
  x_grid.delta_inv = 1.0 / x_grid.delta;
  spline->x_grid   = x_grid;

  if (yBC.lCode == PERIODIC || yBC.lCode == ANTIPERIODIC)
    Ny = My + 3;
  else
    Ny = My + 2;
  y_grid.delta     = (y_grid.end - y_grid.start) / (double)(Ny - 3);
  y_grid.delta_inv = 1.0 / y_grid.delta;
  spline->y_grid   = y_grid;

  if (zBC.lCode == PERIODIC || zBC.lCode == ANTIPERIODIC)
    Nz = Mz + 3;
  else
    Nz = Mz + 2;
  z_grid.delta     = (z_grid.end - z_grid.start) / (double)(Nz - 3);
  z_grid.delta_inv = 1.0 / z_grid.delta;
  spline->z_grid   = z_grid;

  const int ND   = QMC_CLINE / sizeof(double);
  const size_t N = ((num_splines + ND - 1) / ND) * ND;

  spline->x_stride = (size_t)Ny * (size_t)Nz * N;
  spline->y_stride = Nz * N;
  spline->z_stride = N;

  spline->coefs_size = (size_t)Nx * spline->x_stride;

  einspline_create_multi_UBspline_3d_d_coefs<D>(spline, Nx, Ny, Nz, N);

  if (!spline->coefs)
  {
    fprintf(stderr,
            "Out of memory allocating spline coefficients in "
            "create_multi_UBspline_3d_d.\n");
    abort();
  }
}

template<Devices D>
void einspline_create_UBspline_3d_d(UBspline_3d_d<D>*& spline, Ugrid x_grid, Ugrid y_grid,
                                    Ugrid z_grid, BCtype_d xBC, BCtype_d yBC, BCtype_d zBC)
{
  // Create new spline
  spline         = (UBspline_3d_d<D>*)malloc(sizeof(UBspline_3d_d<D>));
  spline->spcode = U3D;
  spline->tcode  = DOUBLE_REAL;
  spline->xBC    = xBC;
  spline->yBC    = yBC;
  spline->zBC    = zBC;

  // Setup internal variables
  int Mx = x_grid.num;
  int My = y_grid.num;
  int Mz = z_grid.num;
  int Nx, Ny, Nz;

  if (xBC.lCode == PERIODIC || xBC.lCode == ANTIPERIODIC)
    Nx = Mx + 3;
  else
    Nx = Mx + 2;
  x_grid.delta     = (x_grid.end - x_grid.start) / (double)(Nx - 3);
  x_grid.delta_inv = 1.0 / x_grid.delta;
  spline->x_grid   = x_grid;

  if (yBC.lCode == PERIODIC || yBC.lCode == ANTIPERIODIC)
    Ny = My + 3;
  else
    Ny = My + 2;
  y_grid.delta     = (y_grid.end - y_grid.start) / (double)(Ny - 3);
  y_grid.delta_inv = 1.0 / y_grid.delta;
  spline->y_grid   = y_grid;

  if (zBC.lCode == PERIODIC || zBC.lCode == ANTIPERIODIC)
    Nz = Mz + 3;
  else
    Nz = Mz + 2;
  z_grid.delta     = (z_grid.end - z_grid.start) / (double)(Nz - 3);
  z_grid.delta_inv = 1.0 / z_grid.delta;
  spline->z_grid   = z_grid;

  spline->x_stride = Ny * Nz;
  spline->y_stride = Nz;

  spline->coefs_size = (size_t)Nx * (size_t)Ny * (size_t)Nz;
  einspline_create_UBspline_3d_d_coefs<D>(spline, Nx, Ny, Nz);
}

template<Devices D>
void einspline_create_UBspline_3d_s(UBspline_3d_s<D>*& spline, Ugrid x_grid, Ugrid y_grid,
                                    Ugrid z_grid, BCtype_s xBC, BCtype_s yBC, BCtype_s zBC)
{
  // Create new spline
  spline         = (UBspline_3d_s<D>*)malloc(sizeof(UBspline_3d_s<D>));
  spline->spcode = U3D;
  spline->tcode  = SINGLE_REAL;
  spline->xBC    = xBC;
  spline->yBC    = yBC;
  spline->zBC    = zBC;
  // Setup internal variables
  int Mx = x_grid.num;
  int My = y_grid.num;
  int Mz = z_grid.num;
  int Nx, Ny, Nz;

  if (xBC.lCode == PERIODIC || xBC.lCode == ANTIPERIODIC)
    Nx = Mx + 3;
  else
    Nx = Mx + 2;
  x_grid.delta     = (x_grid.end - x_grid.start) / (double)(Nx - 3);
  x_grid.delta_inv = 1.0 / x_grid.delta;
  spline->x_grid   = x_grid;

  if (yBC.lCode == PERIODIC || yBC.lCode == ANTIPERIODIC)
    Ny = My + 3;
  else
    Ny = My + 2;
  y_grid.delta     = (y_grid.end - y_grid.start) / (double)(Ny - 3);
  y_grid.delta_inv = 1.0 / y_grid.delta;
  spline->y_grid   = y_grid;

  if (zBC.lCode == PERIODIC || zBC.lCode == ANTIPERIODIC)
    Nz = Mz + 3;
  else
    Nz = Mz + 2;
  z_grid.delta     = (z_grid.end - z_grid.start) / (double)(Nz - 3);
  z_grid.delta_inv = 1.0 / z_grid.delta;
  spline->z_grid   = z_grid;

  spline->x_stride = Ny * Nz;
  spline->y_stride = Nz;

  spline->coefs_size = (size_t)Nx * (size_t)Ny * (size_t)Nz;
  einspline_create_UBspline_3d_s_coefs<D>(spline, Nx, Ny, Nz);
}

// This is necessary boilerplate to avoid unecessary multiple compilation.
// Looking into metaprogramming solution or refactor that eleminates the need
// for so much explicit instantiation

template void einspline_create_multi_UBspline_3d_s(multi_UBspline_3d_s<Devices::CPU>*& spline,
                                                   Ugrid x_grid, Ugrid y_grid, Ugrid z_grid,
                                                   BCtype_s xBC, BCtype_s yBC, BCtype_s zBC,
                                                   int num_splines);

template void einspline_create_multi_UBspline_3d_d(multi_UBspline_3d_d<Devices::CPU>*& spline,
                                                   Ugrid x_grid, Ugrid y_grid, Ugrid z_grid,
                                                   BCtype_d xBC, BCtype_d yBC, BCtype_d zBC,
                                                   int num_splines);

template void einspline_create_UBspline_3d_s(UBspline_3d_s<Devices::CPU>*& spline, Ugrid x_grid,
                                             Ugrid y_grid, Ugrid z_grid, BCtype_s xBC, BCtype_s yBC,
                                             BCtype_s zBC);

template void einspline_create_UBspline_3d_d(UBspline_3d_d<Devices::CPU>*& spline, Ugrid x_grid,
                                             Ugrid y_grid, Ugrid z_grid, BCtype_d xBC, BCtype_d yBC,
                                             BCtype_d zBC);

#ifdef QMC_USE_KOKKOS
template void
einspline_create_multi_UBspline_3d_s(multi_UBspline_3d_s<Devices::KOKKOS>*& restrict spline,
                                     Ugrid x_grid, Ugrid y_grid, Ugrid z_grid, BCtype_s xBC,
                                     BCtype_s yBC, BCtype_s zBC, int num_splines);

template void einspline_create_multi_UBspline_3d_d(multi_UBspline_3d_d<Devices::KOKKOS>*& spline,
                                                   Ugrid x_grid, Ugrid y_grid, Ugrid z_grid,
                                                   BCtype_d xBC, BCtype_d yBC, BCtype_d zBC,
                                                   int num_splines);
template void einspline_create_UBspline_3d_s(UBspline_3d_s<Devices::KOKKOS>*& spline, Ugrid x_grid,
                                             Ugrid y_grid, Ugrid z_grid, BCtype_s xBC, BCtype_s yBC,
                                             BCtype_s zBC);


template void einspline_create_UBspline_3d_d(UBspline_3d_d<Devices::KOKKOS>*& spline, Ugrid x_grid,
                                             Ugrid y_grid, Ugrid z_grid, BCtype_d xBC, BCtype_d yBC,
                                             BCtype_d zBC);

#endif
// #ifdef QMC_USE_CUDA
// template void
// einspline_create_multi_UBspline_3d_s(multi_UBspline_3d_s<Devices::CUDA>*& restrict spline,
//                                      Ugrid x_grid, Ugrid y_grid, Ugrid z_grid, BCtype_s xBC,
//                                      BCtype_s yBC, BCtype_s zBC, int num_splines);

// template void einspline_create_multi_UBspline_3d_d(multi_UBspline_3d_d<Devices::CUDA>*& spline,
//                                                    Ugrid x_grid, Ugrid y_grid, Ugrid z_grid,
//                                                    BCtype_d xBC, BCtype_d yBC, BCtype_d zBC,
//                                                    int num_splines);
// template void einspline_create_UBspline_3d_s(UBspline_3d_s<Devices::CUDA>*& spline, Ugrid x_grid,
//                                              Ugrid y_grid, Ugrid z_grid, BCtype_s xBC, BCtype_s yBC,
//                                              BCtype_s zBC);


// template void einspline_create_UBspline_3d_d(UBspline_3d_d<Devices::CUDA>*& spline, Ugrid x_grid,
//                                              Ugrid y_grid, Ugrid z_grid, BCtype_d xBC, BCtype_d yBC,
//                                              BCtype_d zBC);

// #endif
