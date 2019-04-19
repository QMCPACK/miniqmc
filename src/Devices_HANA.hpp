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

#include <boost/hana.hpp>
// /map.hpp>
// #include <boost/hana/range.hpp>
// #include <boost/hana/tuple.hpp>
// #include <boost/hana/size.hpp>
// #include <boost/hana/fwd/drop_while.hpp>
// #include <boost/hana/string.hpp>

/** @file
 *  nvcc can't deal with hana until cuda 10
 *  this file is to keep it from seeing any hana headers
 */
namespace qmcplusplus
{
namespace hana = boost::hana;

//BOOST_HANA_CONSTEXPR_LAMBDA auto cpu_str = BOOST_HANA_STRING("CPU");
//BOOST_HANA_CONSTEXPR_LAMBDA auto cuda_str = BOOST_HANA_STRING("CUDA");
//BOOST_HANA_CONSTEXPR_LAMBDA auto last_str = BOOST_HANA_STRING("LAST");
// Allows expression of Devices enum as strings at compile time.
    constexpr auto device_names = hana::make_tuple(hana::string_c<'C','P','U'>,
#ifdef QMC_USE_KOKKOS
						   hana::string_c<'K','O','K','K','O'.'S'>,
#endif
#ifdef QMC_USE_OMPOL
						   "OMPOL",
#endif
#ifdef QMC_USE_CUDA
						   hana::string_c<'C','U','D','A'>,
#endif
						   hana::string_c<'L','A','S','T'>);


template <typename Iterable, typename T>
constexpr auto index_of(Iterable const& iterable, T const& element) {
    auto size = decltype(hana::size(iterable)){};
    auto dropped = decltype(hana::size(
        hana::drop_while(iterable, hana::not_equal.to(element))
    )){};
    return size - dropped;
}
    
//Compile time integral range of the Devices enum
constexpr auto devices_range =
    hana::make_range(hana::int_c<static_cast<int>(Devices::CPU)>, hana::int_c<static_cast<int>(Devices::LAST)>);
} // namespace qmcplusplus

#endif
