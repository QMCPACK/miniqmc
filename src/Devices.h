////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2018 QMCPACK developers.
//
// File developed by:
// Peter Doak, doakpw@ornl.gov, Oak Ridge National Lab
//
// File created by:
// Peter Doak, doakpw@ornl.gov, Oak Ridge National Lab
////////////////////////////////////////////////////////////////////////////////

#ifndef QMCPLUSPLUS_DEVICES_H
#define QMCPLUSPLUS_DEVICES_H

#include <boost/hana/map.hpp>
#include <boost/hana/range.hpp>
#include <boost/hana/tuple.hpp>

namespace qmcplusplus
{
namespace hana = boost::hana;

struct DefinedCPU
{
  int defined;
};
constexpr DefinedCPU defined_cpu();

struct DefinedKOKKOS
{
#ifdef QMC_USE_KOKKOS
  int defined;
#endif
};
constexpr DefinedKOKKOS defined_kokkos();

struct DefinedOMPOL
{
#ifdef QMC_USE_OMPOL
  int defined;
#endif
};


/** Enum of Devices supported by this build
 *  used as a template parameter for device specializations
 *  and for template metaprogramming.
 *  I think it should be possible to not have to have duplicate
 *  Enum and names code.
 */
enum class Devices
{
  CPU,
#ifdef QMC_USE_KOKKOS
  KOKKOS,
#endif
#ifdef QMC_USE_OMPOL
  OMPOL,
#endif
  LAST // this is the end
};

  
// clang-format off
// struct CTD // Compile Time Devices
// {
//   struct CPU {};
//   static constexpr auto device_types = hana::make_map(hana::make_pair(hana::int_c<static_cast<int>(Devices::CPU)>,
// 				     hana::type_c<CPU>));
// #ifdef QMC_USE_KOKKOS
//   struct KOKKOS {};
//   static constexpr auto device_types = hana::insert(device_types, hana::make_map(hana::make_pair(hana::int_c<static_cast<int>(Devices::CPU)>,
// 							    hana::type_c<KOKKOS>));
// #endif
// #ifdef QMC_USE_OMPOL
//   struct OMPOL {};
//   static constexpr auto device_types = hana::insert(device_types, hana::make_map(hana::make_pair(hana::int_c<static_cast<int>(Devices::KOKKOS)>,
// 							    hana::type_c<OMPOL>));
// #endif
// }; // namespace Devices_ct
// clang-format on

 
// constexpr auto hana_devices = hana::tuple(Devices);
// Allows expression of Devices enum as strings at compile time.
constexpr auto device_names = hana::make_tuple("CPU",
#ifdef QMC_USE_KOKKOS
					       "KOKKOS",
#endif
#ifdef QMC_USE_OMPOL
					       "OMPOL",
#endif
					       "LAST");

//Compile time integral range of the Devices enum
constexpr auto devices_range = hana::make_range(hana::int_c<static_cast<int>(Devices::CPU)>,
					    hana::int_c<static_cast<int>(Devices::LAST)>);
} // namespace qmcplusplus


#endif
