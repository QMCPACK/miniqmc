#include <numeric>
#include <cstring>
#include "Devices.h"
#include "Numerics/Spline2/MultiBsplineFuncs.hpp"
#include "Numerics/Spline2/MultiBsplineFuncsCUDA.hpp"
#include "Numerics/Spline2/BsplineAllocatorCUDA.hpp"
#include "Numerics/Einspline/MultiBsplineCreateCUDA.h"


namespace qmcplusplus
{
namespace einspline
{
// template<typename ST>
// void Allocator<Devices::CUDA>::destroy(ST* spline)
// {}

template<typename T, typename DT>
struct MBspline_create_cuda;

template<>
struct MBspline_create_cuda<float, float>
{
  typename bspline_traits<Devices::CUDA, float, 3>::SplineType*
  operator()(typename bspline_traits<Devices::CPU, float, 3>::SplineType*& spline)
  {
    return create_multi_UBspline_3d_s_cuda(spline);
  }
};

template<>
struct MBspline_create_cuda<double, float>
{
  typename bspline_traits<Devices::CUDA, float, 3>::SplineType*
  operator()(typename bspline_traits<Devices::CPU, double, 3>::SplineType*& spline)
  {
    return create_multi_UBspline_3d_s_cuda_conv(spline);
  }
};

template<>
struct MBspline_create_cuda<double, double>
{
  typename bspline_traits<Devices::CUDA, double, 3>::SplineType*
  operator()(typename bspline_traits<Devices::CPU, double, 3>::SplineType*& spline)
  {
    return create_multi_UBspline_3d_d_cuda(spline);
  }
};

template<typename T>
struct einspline_cpu_coefs_create;

template<>
struct einspline_cpu_coefs_create<float>
{
  void
  operator()(typename bspline_traits<Devices::CPU, float, 3>::SplineType*& spline, int num_splines)
  {
    einspline_create_multi_UBspline_3d_s_coefs<Devices::CPU>(spline, spline->x_grid.num + 3,
                                                             spline->y_grid.num + 3,
                                                             spline->z_grid.num + 3, num_splines);
  }
};

template<>
struct einspline_cpu_coefs_create<double>
{
  void
  operator()(typename bspline_traits<Devices::CPU, double, 3>::SplineType*& spline, int num_splines)
  {
    einspline_create_multi_UBspline_3d_d_coefs<Devices::CPU>(spline, spline->x_grid.num + 3,
                                                             spline->y_grid.num + 3,
                                                             spline->z_grid.num + 3, num_splines);
  }
};


template<>
void Allocator<Devices::CUDA>::createMultiBspline(
    typename bspline_traits<Devices::CPU, float, 3>::SplineType*& spline,
    typename bspline_traits<Devices::CUDA, float, 3>::SplineType*& target_spline, float dummyT,
    float dummDT)
{
  target_spline = create_multi_UBspline_3d_s_cuda(spline);
}

template<>
void Allocator<Devices::CUDA>::createMultiBspline(
    typename bspline_traits<Devices::CPU, double, 3>::SplineType*& spline,
    typename bspline_traits<Devices::CUDA, double, 3>::SplineType*& target_spline, double dummyT,
    double dummDT)
{
  target_spline = create_multi_UBspline_3d_d_cuda(spline);
}

template<>
void Allocator<Devices::CUDA>::createMultiBspline(
    typename bspline_traits<Devices::CPU, double, 3>::SplineType*& spline,
    typename bspline_traits<Devices::CUDA, float, 3>::SplineType*& target_spline, double dummyT,
    float dummyDT)
{
  target_spline = create_multi_UBspline_3d_s_cuda_conv(spline);
}

template<typename T, typename DT>
void Allocator<Devices::CUDA>::createMultiBspline(
    aligned_vector<typename bspline_traits<Devices::CPU, T, 3>::SplineType*>& cpu_splines,
    typename bspline_traits<Devices::CUDA, DT, 3>::SplineType*& target_spline, T dummyT, DT dummyDT)
{
  int num_splines =
      std::accumulate(cpu_splines.begin(), cpu_splines.end(), 0,
                      [](int a, typename bspline_traits<Devices::CPU, T, 3>::SplineType* spl) {
                        return a + spl->num_splines;
                      });
  Allocator<Devices::CPU> cpu_allocator;
  typename bspline_traits<Devices::CPU, T, 3>::SplineType* cpu_spline = cpu_splines[0];
  typename bspline_traits<Devices::CPU, T, 3>::SplineType* spline;
  cpu_allocator.allocateMultiBspline(spline, cpu_spline->x_grid, cpu_spline->y_grid,
                                     cpu_spline->z_grid, cpu_spline->xBC, cpu_spline->yBC,
                                     cpu_spline->zBC, num_splines);
  int grid_size =
      (cpu_spline->x_grid.num + cpu_spline->y_grid.num + cpu_spline->z_grid.num) * sizeof(T);
  int num_blocks = cpu_splines.size();
  for (int i = 0; i < num_blocks; ++i)
    std::memcpy(&(spline->coefs[grid_size * i]), (cpu_splines[i]->coefs), grid_size);
  target_spline = MBspline_create_cuda<T, DT>()(spline);
}

template<>
void Allocator<Devices::CUDA>::destroy(multi_UBspline_3d_s<Devices::CUDA>*& spline)
{
  cudaFree(spline->coefs);
  delete spline;
  spline = nullptr;
}

template<>
void Allocator<Devices::CUDA>::destroy(multi_UBspline_3d_d<Devices::CUDA>*& spline)
{
  cudaFree(spline->coefs);
  delete spline;
  spline = nullptr;
}


template class Allocator<qmcplusplus::Devices::CUDA>;
template void Allocator<qmcplusplus::Devices::CUDA>::createMultiBspline(
    aligned_vector<typename bspline_traits<Devices::CPU, double, 3>::SplineType*>& cpu_splines,
    typename bspline_traits<Devices::CUDA, double, 3>::SplineType*& target_spline, double dummyT,
    double dummyDT);

template void Allocator<qmcplusplus::Devices::CUDA>::createMultiBspline(
    aligned_vector<typename bspline_traits<Devices::CPU, double, 3>::SplineType*>& cpu_splines,
    typename bspline_traits<Devices::CUDA, float, 3>::SplineType*& target_spline, double dummyT,
    float dummyDT);

template void Allocator<qmcplusplus::Devices::CUDA>::createMultiBspline(
    aligned_vector<typename bspline_traits<Devices::CPU, float, 3>::SplineType*>& cpu_splines,
    typename bspline_traits<Devices::CUDA, float, 3>::SplineType*& target_spline, float dummyT,
    float dummyDT);

} // namespace einspline
} // namespace qmcplusplus
