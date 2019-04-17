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

#ifndef QMCPLUSPLUS_DEVICES_HANA_H
#define QMCPLUSPLUS_DEVICES_HANA_H

#include <boost/hana/map.hpp>
#include <boost/hana/range.hpp>
#include <boost/hana/tuple.hpp>

/** @file
 *  nvcc can't deal with hana until cuda 10
 *  this file is to keep it from seeing any hana headers
 */
namespace qmcplusplus
{
namespace hana = boost::hana;

// Allows expression of Devices enum as strings at compile time.
constexpr auto device_names = hana::make_tuple("CPU",
#ifdef QMC_USE_KOKKOS
                                               "KOKKOS",
#endif
#ifdef QMC_USE_OMPOL
                                               "OMPOL",
#endif
#ifdef QMC_USE_CUDA
                                               "CUDA",
#endif
                                               "LAST");

//Compile time integral range of the Devices enum
constexpr auto devices_range =
    hana::make_range(hana::int_c<static_cast<int>(Devices::CPU)>, hana::int_c<static_cast<int>(Devices::LAST)>);
} // namespace qmcplusplus

#endif
