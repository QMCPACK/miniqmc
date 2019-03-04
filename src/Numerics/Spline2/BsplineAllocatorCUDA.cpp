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


template<>
void Allocator<Devices::CUDA>::createMultiBspline(typename bspline_traits<Devices::CPU, float, 3>::SplineType*& spline,
						     typename bspline_traits<Devices::CUDA, float, 3>::SplineType*& target_spline, float dummyT, float dummDT)
{
  target_spline = create_multi_UBspline_3d_s_cuda(spline);
}

template<>
void Allocator<Devices::CUDA>::createMultiBspline(typename bspline_traits<Devices::CPU, double, 3>::SplineType*& spline,
						     typename bspline_traits<Devices::CUDA, double, 3>::SplineType*& target_spline, double dummyT, double dummDT)
{
  target_spline = create_multi_UBspline_3d_d_cuda(spline);
}

template<>
void Allocator<Devices::CUDA>::createMultiBspline(typename bspline_traits<Devices::CPU, double, 3>::SplineType*& spline,
						     typename bspline_traits<Devices::CUDA, float, 3>::SplineType*& target_spline, double dummyT, float dummyDT)
{
  target_spline = create_multi_UBspline_3d_s_cuda_conv(spline);
}

template<>
void Allocator<Devices::CUDA>::destroy(multi_UBspline_3d_s<Devices::CUDA>*& spline)
{
  cudaFree(spline->coefs);
  free(spline);
  spline = nullptr;
}

template<>
void Allocator<Devices::CUDA>::destroy(multi_UBspline_3d_d<Devices::CUDA>*& spline)
{
  cudaFree(spline->coefs);
  free(spline);
  spline = nullptr;
}

    

template class Allocator<qmcplusplus::Devices::CUDA>;
}
}
