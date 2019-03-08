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
/** @file bspline_allocator.hpp
 * @brief Allocator and management classes
 */
#ifndef QMCPLUSPLUS_BSPLINE_ALLOCATOR_CUDA_HPP
#define QMCPLUSPLUS_BSPLINE_ALLOCATOR_CUDA_HPP

#include "Devices.h"
#include "Utilities/SIMD/allocator.hpp"
#include "Numerics/Spline2/bspline_allocator.hpp"
#include "Numerics/Spline2/bspline_traits.hpp"
#include "Numerics/Spline2/einspline_allocator.h"
#include "Numerics/Einspline/multi_bspline_structs.h"
#include "Numerics/Einspline/multi_bspline_structs_cuda.h"
#include <Numerics/OhmmsPETE/OhmmsArray.h>

namespace qmcplusplus
{
namespace einspline
{

// template<Typename CPU_SPLINE, Typename GPU_SPLINE>
// void createMultiBspline_3d(CPU_SPLINE* spline,
// 			   GPU_SPLINE* target_spline)
// {
//   constexpr bool hana::type_c<CPU_SPLINE>
			   
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

  // /// create a single CUDA multi-bspline
  // void createMultiBspline_3d(multi_UBspline_3d_s<qmcplusplus::Devices::CPU>* source_spline,
  // 			     multi_UBspline_3d_s<Devices::CUDA>* target_spline);

  // void createMultiBspline_3d(multi_UBspline_3d_d<qmcplusplus::Devices::CPU>* source_spline,
  // 			  multi_UBspline_3d_s<Devices::CUDA>* target_spline);

  // /// create a double CUDA multi-bspline
  // void createMultiBspline_3d(multi_UBspline_3d_d<qmcplusplus::Devices::CPU>* source_spline,
  // 			     multi_UBspline_3d_d<Devices::CUDA>* target_spline);
};


extern template class Allocator<qmcplusplus::Devices::CUDA>;
    extern template void Allocator<qmcplusplus::Devices::CUDA>::createMultiBspline(  aligned_vector<typename bspline_traits<Devices::CPU, double, 3>::SplineType*>& cpu_splines,
										      typename bspline_traits<Devices::CUDA, double, 3>::SplineType*& target_spline, double dummyT, double dummyDT);

       extern template void Allocator<qmcplusplus::Devices::CUDA>::createMultiBspline(  aligned_vector<typename bspline_traits<Devices::CPU, double, 3>::SplineType*>& cpu_splines,
										      typename bspline_traits<Devices::CUDA, float, 3>::SplineType*& target_spline, double dummyT, float dummyDT);

        extern template void Allocator<qmcplusplus::Devices::CUDA>::createMultiBspline(  aligned_vector<typename bspline_traits<Devices::CPU, float, 3>::SplineType*>& cpu_splines,
										      typename bspline_traits<Devices::CUDA, float, 3>::SplineType*& target_spline, float dummyT, float dummyDT);

} // namespace einspline
} // namespace qmcplusplus

#endif
