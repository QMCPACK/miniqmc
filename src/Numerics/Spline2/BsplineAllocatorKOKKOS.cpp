#include "Numerics/Spline2/BsplineAllocatorKOKKOS.hpp"
#include "einspline_allocator_kokkos.h"

namespace qmcplusplus
{
namespace einspline
{

//template<>
Allocator<Devices::KOKKOS>::Allocator():Policy(0) {}

//template<>
Allocator<Devices::KOKKOS>::~Allocator() {}

template<typename T, typename ValT, typename IntT>
void Allocator<Devices::KOKKOS>::createMultiBspline(typename bspline_traits<DT, T, 3>::SplineType*& spline, T dummy,
                                      ValT& start, ValT& end, IntT& ng, bc_code bc, int num_splines)
{
  Ugrid x_grid, y_grid, z_grid;
  typename bspline_traits<DT, T, 3>::BCType xBC, yBC, zBC;
  x_grid.start = start[0];
  x_grid.end   = end[0];
  x_grid.num   = ng[0];
  y_grid.start = start[1];
  y_grid.end   = end[1];
  y_grid.num   = ng[1];
  z_grid.start = start[2];
  z_grid.end   = end[2];
  z_grid.num   = ng[2];
  xBC.lCode = xBC.rCode = bc;
  yBC.lCode = yBC.rCode = bc;
  zBC.lCode = zBC.rCode = bc;
  allocateMultiBspline(spline, x_grid, y_grid, z_grid, xBC, yBC, zBC, num_splines);
}

void Allocator<Devices::KOKKOS>::allocateMultiBspline(
  multi_UBspline_3d_d<DT>*& spline, Ugrid x_grid, Ugrid y_grid, Ugrid z_grid, BCtype_d xBC, BCtype_d yBC, BCtype_d zBC, int num_splines)
{
  einspline_create_multi_UBspline_3d_d(spline, x_grid, y_grid, z_grid, xBC, yBC, zBC, num_splines);
}
void Allocator<Devices::KOKKOS>::allocateMultiBspline(
  multi_UBspline_3d_s<DT>*& spline, Ugrid x_grid, Ugrid y_grid, Ugrid z_grid, BCtype_s xBC, BCtype_s yBC, BCtype_s zBC, int num_splines)
{
  einspline_create_multi_UBspline_3d_s(spline, x_grid, y_grid, z_grid, xBC, yBC, zBC, num_splines);
}

template<typename T>
void Allocator<Devices::KOKKOS>::setCoefficientsForOneOrbital(
    int i,
    Kokkos::View<T***>& coeff,
    typename bspline_traits<Devices::KOKKOS, T, 3>::SplineType* spline)
{
  //  #pragma omp parallel for collapse(3)
  for (int ix = 0; ix < spline->x_grid.num + 3; ix++)
  {
    for (int iy = 0; iy < spline->y_grid.num + 3; iy++)
    {
      for (int iz = 0; iz < spline->z_grid.num + 3; iz++)
      {
        intptr_t xs                                    = spline->x_stride;
        intptr_t ys                                    = spline->y_stride;
        intptr_t zs                                    = spline->z_stride;
        spline->coefs[ix * xs + iy * ys + iz * zs + i] = coeff(ix, iy, iz);
      }
    }
  }
}

template<typename SplineType>
void Allocator<Devices::KOKKOS>::destroy(SplineType*& spline)
{
    //Assign coefs_view to empty view because of Kokkos reference counting
    // and garbage collection.
    //spline->coefs_view = multi_UBspline_3d_d::coefs_view_t();
    spline->coefs_view = SplineType::coefs_view_t();
    free(spline);
}

template class Allocator<Devices::KOKKOS>;
template void Allocator<qmcplusplus::Devices::KOKKOS>::createMultiBspline(typename bspline_traits<Devices::KOKKOS, double, 3u>::SplineType*& spline, double dummy,
										     TinyVector<double, 3u>& start, TinyVector<double, 3u>& end, TinyVector<int, 3u>& ng, bc_code bc, int num_splines);
template void Allocator<Devices::KOKKOS>::setCoefficientsForOneOrbital(
    int i,
    Kokkos::View<double***>& coeff,
    typename bspline_traits<Devices::KOKKOS, double, 3>::SplineType* spline);

}
}
