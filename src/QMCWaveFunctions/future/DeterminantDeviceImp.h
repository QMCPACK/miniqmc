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

#ifndef QMCPLUSPLUS_FUTURE_DETERMINANT_DEVICE_IMP_H
#define QMCPLUSPLUS_FUTURE_DETERMINANT_DEVICE_IMP_H

// This is the only location where different determinant implmentations
// should be included or not
#include <typeinfo>
#include <ostream>
#include <array>
#include <boost/hana/range.hpp>
#include <boost/hana/tuple.hpp>
#include <boost/hana/string.hpp>
#include <boost/hana/type.hpp>

namespace qmcplusplus
{
namespace future
{
enum class DeterminantDeviceType
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

using DDT = DeterminantDeviceType;  
namespace hana = boost::hana;
  
constexpr auto ddt_names = hana::make_tuple("CPU",
#ifdef QMC_USE_KOKKOS
					    "KOKKOS",
#endif
#ifdef QMC_USE_OMPOL
					    "OMPOL",
#endif
					     "LAST");

  

template<DeterminantDeviceType DT>
class DeterminantDeviceImp;
}
}

#include "QMCWaveFunctions/future/DeterminantDeviceImpCPU.h"
#ifdef QMC_USE_KOKKOS
#include "QMCWaveFunctions/future/DeterminantDeviceImpKOKKOS.h"
#endif

namespace qmcplusplus
{
namespace future
{

  constexpr auto ddt_range = hana::make_range( hana::int_c<static_cast<int>(DDT::CPU)>,
					      hana::int_c<static_cast<int>(DDT::LAST)>);
  
  extern template class DeterminantDeviceImp<DDT::CPU>;
#ifdef QMC_USE_KOKKOS  
  extern template class DeterminantDeviceImp<DDT::KOKKOS>;
#endif
}
}

#endif
