////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2018 QMCPACK developers.
//
// File developed by:
// Peter Doak, doakpw@ornl.gov, Oak Ridge National Lab
//
// File created by:
// Peter Doak, doakpw@ornl.gov, Oak Ridge National Lab
////////////////////////////////////////////////////////////////////////////////

#include "Devices.h"
#include "Numerics/Spline2/einspline_allocator_kokkos.h"
#include "Numerics/Einspline/multi_bspline_structs_kokkos.h"

// template<>
// void einspline_create_UBspline_3d_d(UBspline_3d_d<Devices::KOKKOS>*& spline,
//                                                     Ugrid x_grid, Ugrid y_grid, Ugrid z_grid,
//                                                     BCtype_d xBC, BCtype_d yBC, BCtype_d zBC)
// {}

// template<>
// void einspline_create_UBspline_3d_s(UBspline_3d_s<Devices::KOKKOS>*& spline,
//                                                     Ugrid x_grid, Ugrid y_grid, Ugrid z_grid,
//                                                     BCtype_s xBC, BCtype_s yBC, BCtype_s zBC)
// {}

template<>
void einspline_create_multi_UBspline_3d_s(multi_UBspline_3d_s<Devices::KOKKOS>*& restrict spline,
                                          Ugrid x_grid,
                                          Ugrid y_grid,
                                          Ugrid z_grid,
                                          BCtype_s xBC,
                                          BCtype_s yBC,
                                          BCtype_s zBC,
                                          int num_splines)
{
  // Create new spline
  spline = new multi_UBspline_3d_s<Devices::KOKKOS>;
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
  int N            = (num_splines % ND) ? (num_splines + ND - num_splines % ND) : num_splines;
  spline->x_stride = (size_t)Ny * (size_t)Nz * (size_t)N;
  spline->y_stride = (size_t)Nz * N;
  spline->z_stride = N;

  spline->coefs_size = (size_t)Nx * spline->x_stride;

  spline->coefs_view =
      multi_UBspline_3d_s<Devices::KOKKOS>::coefs_view_t("Multi_UBspline_3d_s", Nx, Ny, Nz, N);


  //Check that data layout is as expected
  //
  int strides[4];
  spline->coefs_view.stride(strides);
  if (spline->x_stride != strides[0] || spline->y_stride != strides[1] ||
      spline->z_stride != strides[2] || 1 != strides[3])
    fprintf(stderr,
            "Kokkos View has non-compatible strides %i %i | %i %i | %i %i\n",
            spline->x_stride,
            strides[0],
            spline->y_stride,
            strides[1],
            spline->z_stride,
            strides[2]);

  spline->coefs = spline->coefs_view.data();

  if (!spline->coefs)
  {
    fprintf(stderr,
            "Out of memory allocating spline coefficients in "
            "create_multi_UBspline_3d_s.\n");
    abort();
  }
}

template<>
void einspline_create_multi_UBspline_3d_d(multi_UBspline_3d_d<Devices::KOKKOS>*& spline,
                                          Ugrid x_grid,
                                          Ugrid y_grid,
                                          Ugrid z_grid,
                                          BCtype_d xBC,
                                          BCtype_d yBC,
                                          BCtype_d zBC,
                                          int num_splines)
{
  // Create new spline
  spline = new multi_UBspline_3d_d<Devices::KOKKOS>;
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

  const int ND     = QMC_CLINE / sizeof(double);
  int N            = (num_splines % ND) ? (num_splines + ND - num_splines % ND) : num_splines;
  spline->x_stride = (size_t)Ny * (size_t)Nz * (size_t)N;
  spline->y_stride = (size_t)Nz * N;
  spline->z_stride = N;

  spline->coefs_size = (size_t)Nx * spline->x_stride;

  spline->coefs_view =
      multi_UBspline_3d_d<Devices::KOKKOS>::coefs_view_t("Multi_UBspline_3d_s", Nx, Ny, Nz, N);


  //Check that data layout is as expected
  //
  int strides[4];
  spline->coefs_view.stride(strides);
  if (spline->x_stride != strides[0] || spline->y_stride != strides[1] ||
      spline->z_stride != strides[2] || 1 != strides[3])
    fprintf(stderr,
            "Kokkos View has non-compatible strides %i %i | %i %i | %i %i\n",
            spline->x_stride,
            strides[0],
            spline->y_stride,
            strides[1],
            spline->z_stride,
            strides[2]);

  spline->coefs = spline->coefs_view.data();

  if (!spline->coefs)
  {
    fprintf(stderr,
            "Out of memory allocating spline coefficients in "
            "create_multi_UBspline_3d_s.\n");
    abort();
  }
}


template<>
void einspline_create_multi_UBspline_3d_s_coefs(
    multi_UBspline_3d_s<Devices::KOKKOS>*& restrict spline, int Nx, int Ny, int Nz, int N)
{
  spline->coefs_view =
      multi_UBspline_3d_s<Devices::KOKKOS>::coefs_view_t("Multi_UBspline_3d_s", Nx, Ny, Nz, N);


  //Check that data layout is as expected
  //
  int strides[4];
  spline->coefs_view.stride(strides);
  if (spline->x_stride != strides[0] || spline->y_stride != strides[1] ||
      spline->z_stride != strides[2] || 1 != strides[3])
    fprintf(stderr,
            "Kokkos View has non-compatible strides %ld %i | %ld %i | %ld %i\n",
            spline->x_stride,
            strides[0],
            spline->y_stride,
            strides[1],
            spline->z_stride,
            strides[2]);

  spline->coefs = spline->coefs_view.data();
}

template<>
void einspline_create_multi_UBspline_3d_d_coefs(
    multi_UBspline_3d_d<Devices::KOKKOS>*& restrict spline, int Nx, int Ny, int Nz, int N)
{
  spline->coefs_view =
      multi_UBspline_3d_d<Devices::KOKKOS>::coefs_view_t("Multi_UBspline_3d_d", Nx, Ny, Nz, N);

  //Check that data layout is as expected
  //
  int strides[4];
  spline->coefs_view.stride(strides);
  if (spline->x_stride != strides[0] || spline->y_stride != strides[1] ||
      spline->z_stride != strides[2] || 1 != strides[3])
    fprintf(stderr,
            "Kokkos View has non-compatible strides %ld %i | %ld %i | %ld %i\n",
            spline->x_stride,
            strides[0],
            spline->y_stride,
            strides[1],
            spline->z_stride,
            strides[2]);

  spline->coefs = spline->coefs_view.data();
}
