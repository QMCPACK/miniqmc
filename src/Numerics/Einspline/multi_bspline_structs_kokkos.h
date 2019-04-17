/////////////////////////////////////////////////////////////////////////////
//  einspline:  a library for creating and evaluating B-splines            //
//  Modified Peter Doak 2019
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

#ifndef MULTI_BSPLINE_STRUCTS_KOKKOS_H
#define MULTI_BSPLINE_STRUCTS_KOKKOS_H

#include "multi_bspline_structs.h"

#define MULTI_UBSPLINE_KOKKOS_VIEW_DEF                                             \
  typedef Kokkos::View<KokkosViewPrecision****, Kokkos::LayoutRight> coefs_view_t; \
  coefs_view_t coefs_view;

template<>
struct multi_UBspline_3d_s<Devices::KOKKOS> : public multi_UBspline_3d_s_common
{
  using KokkosViewPrecision = float;
  MULTI_UBSPLINE_KOKKOS_VIEW_DEF
};

template<>
struct multi_UBspline_3d_d<Devices::KOKKOS> : public multi_UBspline_3d_d_common
{
  using KokkosViewPrecision = double;
  MULTI_UBSPLINE_KOKKOS_VIEW_DEF
};

#endif
