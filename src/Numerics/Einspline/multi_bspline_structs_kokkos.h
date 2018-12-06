#ifndef MULTI_BSPLINE_STRUCTS_KOKKOS_H
#define MULTI_BSPLINE_STRUCTS_KOKKOS_H

#include "multi_bspline_structs.h"

#define MULTI_UBSPLINE_KOKKOS_VIEW_DEF \
  typedef Kokkos::View<KokkosViewPrecision****, Kokkos::LayoutRight> coefs_view_t;\
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
