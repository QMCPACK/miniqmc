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

#ifndef QMCPLUSPLUS_DETERMINANT_DEVICE_IMP_H
#define QMCPLUSPLUS_DETERMINANT_DEVICE_IMP_H

#include "Devices.h" 

/** @file
 * Here compiled determinant implmentations
 * are included.
 */

namespace qmcplusplus
{
template<Devices DT>
class DeterminantDeviceImp
{};
}

#include "QMCWaveFunctions/DeterminantDeviceImpCPU.h"
#ifdef QMC_USE_KOKKOS
#include "QMCWaveFunctions/DeterminantDeviceImpKOKKOS.h"
#endif

namespace qmcplusplus
{
}

#endif
