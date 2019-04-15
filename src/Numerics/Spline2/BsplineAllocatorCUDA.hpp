////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by:
//
// File created by: Jeongnim Kim, jeongnim.kim@intel.com, Intel Corp.
////////////////////////////////////////////////////////////////////////////////
// -*- C++ -*-
/** @file BsplineAllocator.hpp
 * @brief Allocator and management classes
 */
#ifndef QMCPLUSPLUS_BSPLINE_ALLOCATOR_CUDA_HPP
#define QMCPLUSPLUS_BSPLINE_ALLOCATOR_CUDA_HPP

#include "Devices.h"
#include "Utilities/SIMD/allocator.hpp"
#include "Numerics/Spline2/BsplineAllocator.hpp"
#include "Numerics/Spline2/bspline_traits.hpp"
#include "Numerics/Spline2/einspline_allocator.h"
#include "Numerics/Einspline/multi_bspline_structs.h"
#include "Numerics/Einspline/multi_bspline_structs_cuda.h"
#include "Numerics/Einspline/MultiBsplineCreateCUDA.h"
#include "Numerics/OhmmsPETE/OhmmsArray.h"

namespace qmcplusplus
{
namespace einspline
{

template<typename T, typename DT>
struct MBspline_create_cuda;

template<>
struct MBspline_create_cuda<float, float>
{
  typename bspline_traits<Devices::CUDA, float, 3>::SplineType*
  operator()(typename bspline_traits<Devices::CPU, float, 3>::SplineType*& spline, int spline_block_size = 0)
  {
    if (spline_block_size == 0)
      spline_block_size = spline->num_splines;
    return create_multi_UBspline_3d_s_cuda(spline, spline_block_size);
  }
};

template<>
struct MBspline_create_cuda<double, float>
{
  typename bspline_traits<Devices::CUDA, float, 3>::SplineType*
  operator()(typename bspline_traits<Devices::CPU, double, 3>::SplineType*& spline, int spline_block_size = 0)
  {
    if (spline_block_size == 0)
      spline_block_size = spline->num_splines;
    return create_multi_UBspline_3d_s_cuda_conv(spline,spline_block_size);
  }
};

template<>
struct MBspline_create_cuda<double, double>
{
  typename bspline_traits<Devices::CUDA, double, 3>::SplineType*
  operator()(typename bspline_traits<Devices::CPU, double, 3>::SplineType*& spline, int spline_block_size = 0)
  {
    if (spline_block_size == 0)
      spline_block_size = spline->num_splines;
    return create_multi_UBspline_3d_d_cuda(spline, spline_block_size);
  }
};


/** CUDA spline allocator requires CPU spline input
 */
template<>
class Allocator<qmcplusplus::Devices::CUDA>
{
  /// Setting the allocation policy: default is using aligned allocator
  int Policy;

public:
  /// constructor
  Allocator() {}
  /// enable default copy constructor
  Allocator(const Allocator&) = default;
  /// disable assignement
  Allocator& operator=(const Allocator&) = delete;
  /// destructor
  ~Allocator() {}

  template<typename SplineType>
  void destroy(SplineType*& spline);

  template<typename T, typename DT>
  void createMultiBspline(typename bspline_traits<Devices::CPU, T, 3>::SplineType*& spline,
			     typename bspline_traits<Devices::CUDA, DT, 3>::SplineType*& target_spline,
			     T dummyT,
			     DT dummyDT);

  template<typename T, typename DT>
  void createMultiBspline(aligned_vector<typename bspline_traits<Devices::CPU, T, 3>::SplineType*>& cpu_splines,
						     typename bspline_traits<Devices::CUDA, DT, 3>::SplineType*& target_spline, T dummyT, DT dummyDT);

  template<typename T>
  void setCoefficientsForOneOrbital(int i,
                                    Array<T, 3>& coeff,
                                    typename bspline_traits<Devices::CUDA, T, 3>::SplineType*& spline)
  {
    assert("This method does nothing and should not be called on BsplineAllocatorCUDA.");
  }

private:
  template<typename T>
  std::vector<int>  extractCPUSplineCounts(const aligned_vector<typename bspline_traits<Devices::CPU, T, 3>::SplineType*>& cpu_splines) const;


  
  // /// create a single CUDA multi-bspline
  // void createMultiBspline_3d(multi_UBspline_3d_s<qmcplusplus::Devices::CPU>* source_spline,
  // 			     multi_UBspline_3d_s<Devices::CUDA>* target_spline);

  // void createMultiBspline_3d(multi_UBspline_3d_d<qmcplusplus::Devices::CPU>* source_spline,
  // 			  multi_UBspline_3d_s<Devices::CUDA>* target_spline);

  // /// create a double CUDA multi-bspline
  // void createMultiBspline_3d(multi_UBspline_3d_d<qmcplusplus::Devices::CPU>* source_spline,
  // 			     multi_UBspline_3d_d<Devices::CUDA>* target_spline);
};

template<typename CPU_T, typename DEV_T>
void Allocator<Devices::CUDA>::createMultiBspline(
    typename bspline_traits<Devices::CPU, CPU_T, 3>::SplineType*& spline,
    typename bspline_traits<Devices::CUDA, DEV_T, 3>::SplineType*& target_spline, CPU_T dummyT,
    DEV_T dummDT)
{
  // using CPU_T = typename bspline_type<decltype(*spline)>::value_type;
  // using DEV_T = typename bspline_type<decltype(*target_spline)>::value_type;
  target_spline = MBspline_create_cuda<CPU_T, DEV_T>()(spline);
  //create_multi_UBspline_3d_s_cuda(spline);
}


extern template class Allocator<qmcplusplus::Devices::CUDA>;
extern template void Allocator<qmcplusplus::Devices::CUDA>::createMultiBspline<double,double>(  aligned_vector<typename bspline_traits<Devices::CPU, double, 3>::SplineType*>& cpu_splines,
    										      typename bspline_traits<Devices::CUDA, double, 3>::SplineType*& target_spline, double dummyT, double dummyDT);

extern template void Allocator<qmcplusplus::Devices::CUDA>::createMultiBspline<double, float>(  aligned_vector<typename bspline_traits<Devices::CPU, double, 3>::SplineType*>& cpu_splines,
    										      typename bspline_traits<Devices::CUDA, float, 3>::SplineType*& target_spline, double dummyT, float dummyDT);

extern template void Allocator<qmcplusplus::Devices::CUDA>::createMultiBspline<float, float>(  aligned_vector<typename bspline_traits<Devices::CPU, float, 3>::SplineType*>& cpu_splines,
    										      typename bspline_traits<Devices::CUDA, float, 3>::SplineType*& target_spline, float dummyT, float dummyDT);

} // namespace einspline
} // namespace qmcplusplus

#endif
