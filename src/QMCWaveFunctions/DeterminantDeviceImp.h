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
 * Here determinant implmentations
 * are included.
 */

namespace qmcplusplus
{
/**
 * Collects device implementations into class templates rather
 * than separate classes.
 * This faciliates other generic code using the device enumeration.
 */
template<Devices DT>
class DeterminantDeviceImp
{};
}

#include "QMCWaveFunctions/DeterminantDeviceImpCPU.h"
#ifdef QMC_USE_KOKKOS
#include "QMCWaveFunctions/DeterminantDeviceImpKOKKOS.h"
#endif
#ifdef QMC_USE_CUDA
#include "QMCWaveFunctions/DeterminantDeviceImpCUDA.h"
#endif

#endif
