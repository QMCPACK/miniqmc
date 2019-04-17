#include "Numerics/Spline2/BsplineAllocatorKOKKOS.hpp"

namespace qmcplusplus
{
namespace einspline
{

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
}
}
