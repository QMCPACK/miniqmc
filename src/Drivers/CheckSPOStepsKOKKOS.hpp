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
#ifndef QMCPLUSPLUS_CHECK_SPO_STEPS_KOKKOS_HPP
#define QMCPLUSPLUS_CHECK_SPO_STEPS_KOKKOS_HPP

#include "Drivers/check_spo.h"
#include "Utilities/Configuration.h"
#include "Drivers/CheckSPOSteps.hpp"

namespace qmcplusplus
{
//template<Devices DT>
//class CheckSPOSteps;
    
/** Kokkos functor for custom reduction
 */
extern template void CheckSPOSteps<Devices::KOKKOS>::test(int&, int, qmcplusplus::Tensor<int, 3u> const&, int, int, int, int, int, double);

extern template class CheckSPOSteps<Devices::KOKKOS>;

} // namespace qmcplusplus

#endif

