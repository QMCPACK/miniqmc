#include <numeric>
#include <cstring>
#include <cassert>
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

/** loads the vector with spline count of each cpu_spline we are merging for the CUDA device
 *  Hopefully getting an RVO.
 */
template<typename T>
std::vector<int> Allocator<Devices::CUDA>::extractCPUSplineCounts(
    const aligned_vector<typename bspline_traits<Devices::CPU, T, 3>::SplineType*>& cpu_splines) const
{
  std::vector<int> cpu_mbs_spline_count;
  for (int i = 0; i < cpu_splines.size(); ++i)
  {
    cpu_mbs_spline_count.emplace_back(cpu_splines[i]->num_splines);
  }
  return cpu_mbs_spline_count;
}

/** This create merges CPU splines to create a larger CUDA spline
 *  Should it protect from merging incompatible CPU splines?
 *  Compatible defined as grid and boundary conditions match
 */
template<typename T, typename DT>
void Allocator<Devices::CUDA>::createMultiBspline(
    aligned_vector<typename bspline_traits<Devices::CPU, T, 3>::SplineType*>& cpu_splines,
    typename bspline_traits<Devices::CUDA, DT, 3>::SplineType*& target_spline, T dummyT, DT dummyDT)
{
  if ( cpu_splines.size() == 1 )
    return Allocator<Devices::CUDA>::createMultiBspline(cpu_splines[0], target_spline, dummyT, dummyDT);
  std::vector<int> cpu_mbs_spline_count =
      Allocator<Devices::CUDA>::extractCPUSplineCounts<T>(cpu_splines);
  int num_splines = std::accumulate(cpu_mbs_spline_count.begin(), cpu_mbs_spline_count.end(), 0);
  Allocator<Devices::CPU> cpu_allocator;
  typename bspline_traits<Devices::CPU, T, 3>::SplineType* cpu_spline = cpu_splines[0];
  typename bspline_traits<Devices::CPU, T, 3>::SplineType* merged_spline;
  cpu_allocator.allocateMultiBspline(merged_spline, cpu_spline->x_grid, cpu_spline->y_grid,
                                     cpu_spline->z_grid, cpu_spline->xBC, cpu_spline->yBC,
                                     cpu_spline->zBC, num_splines);
  // right now only PERIODIC is supported
  // Period bounary condition results in three more coeficients per dimension
  assert(cpu_spline->xBC.lCode == bc_code::PERIODIC);
  assert(cpu_spline->yBC.lCode == bc_code::PERIODIC);
  assert(cpu_spline->zBC.lCode == bc_code::PERIODIC);
  int x_grid_actual = cpu_spline->x_grid.num + 3;
  int y_grid_actual = cpu_spline->y_grid.num + 3;
  int z_grid_actual = cpu_spline->z_grid.num + 3;

  int z_stride_dest = num_splines;
  int y_stride_dest = z_stride_dest * z_grid_actual;
  int x_stride_dest = y_stride_dest * y_grid_actual;
  
  // copying into the merged spline
  for (int i = 0; i < x_grid_actual; ++i)
  {
    for (int j = 0; j < y_grid_actual; ++j)
    {
      for (int k = 0; k < z_grid_actual; ++k)
      {
        int curr_spline_shift = 0;
        for (int ispline = 0; ispline < cpu_splines.size(); ++ispline)
        {
	  int z_stride_src = cpu_mbs_spline_count[ispline];
	  int y_stride_src = z_stride_src * z_grid_actual;
	  int x_stride_src = y_stride_src * y_grid_actual;
  
          std::memcpy(merged_spline->coefs + (i * x_stride_dest +  j * y_stride_dest  +
					      k * z_stride_dest + curr_spline_shift),
                      cpu_splines[ispline]->coefs + (i * x_stride_src + j * y_stride_src +
						     k * z_stride_src),
                      cpu_mbs_spline_count[ispline] * sizeof(T));
          curr_spline_shift += z_stride_src;
        }
      }
    }
  }

  target_spline = MBspline_create_cuda<T, DT>()(merged_spline);
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
