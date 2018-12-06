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

#ifndef EINSPLINE_ALLOCATOR_KOKKOS_H
#define EINSPLINE_ALLOCATOR_KOKKOS_H

extern template void einspline_create_UBspline_3d_d(UBspline_3d_d<Devices::KOKKOS>*& spline,
                                                    Ugrid x_grid, Ugrid y_grid, Ugrid z_grid,
                                                    BCtype_d xBC, BCtype_d yBC, BCtype_d zBC);

extern template void einspline_create_UBspline_3d_s(UBspline_3d_s<Devices::KOKKOS>*& spline,
                                                    Ugrid x_grid, Ugrid y_grid, Ugrid z_grid,
                                                    BCtype_s xBC, BCtype_s yBC, BCtype_s zBC);

extern template void
einspline_create_multi_UBspline_3d_s(multi_UBspline_3d_s<Devices::KOKKOS>*& restrict spline,
                                     Ugrid x_grid, Ugrid y_grid, Ugrid z_grid, BCtype_s xBC,
                                     BCtype_s yBC, BCtype_s zBC, int num_splines);

extern template void
einspline_create_multi_UBspline_3d_d(multi_UBspline_3d_d<Devices::KOKKOS>*& spline, Ugrid x_grid,
                                     Ugrid y_grid, Ugrid z_grid, BCtype_d xBC, BCtype_d yBC,
                                     BCtype_d zBC, int num_splines);

//inlined only due to inclusion in multiple compilation units
template<>
inline void einspline_create_multi_UBspline_3d_s_coefs(multi_UBspline_3d_s<Devices::KOKKOS>*& restrict spline,
						int Nx,
						int Ny,
						int Nz,
						int N)
{
  spline->coefs_view = multi_UBspline_3d_s<Devices::KOKKOS>::coefs_view_t("Multi_UBspline_3d_s", Nx, Ny, Nz, N);


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
inline void einspline_create_multi_UBspline_3d_d_coefs(multi_UBspline_3d_d<Devices::KOKKOS>*& restrict spline,
						int Nx,
						int Ny,
						int Nz,
						int N)
{
  spline->coefs_view = multi_UBspline_3d_d<Devices::KOKKOS>::coefs_view_t("Multi_UBspline_3d_d", Nx, Ny, Nz, N);

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

/* extern template void einspline_create_UBspline_3d_d_coefs(UBspline_3d_d<Devices::KOKKOS>*& spline, */
/* 							  int Nx, int Ny, int Nz, int N); */

/* extern template void einspline_create_UBspline_3d_s_coefs(UBspline_3d_s<Devices::KOKKOS>*& spline, */
/* 							  int Nx, int Ny, int Nz, int N); */
                                                    

#endif
