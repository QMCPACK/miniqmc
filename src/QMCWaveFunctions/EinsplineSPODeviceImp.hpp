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

#ifndef QMCPLUSPLUS_EINSPLINE_SPO_DEVICE_IMP_H
#define QMCPLUSPLUS_EINSPLINE_SPO_DEVICE_IMP_H

#include "Devices.h" 
#include <ostream>
#include "QMCWaveFunctions/EinsplineSPOParams.h"
#include "QMCWaveFunctions/EinsplineSPODevice.hpp"
/** @file
 * Here EinsplineSPO device implementations are included
 */

namespace qmcplusplus
{  
template<Devices DT, typename T>
class EinsplineSPODeviceImp;

template<Devices DT, typename T>
std::ostream& operator<< (std::ostream& os, EinsplineSPODeviceImp<DT, T>& espodi)
{
  const EinsplineSPOParams<T>& e = espodi.getParams();
  os  << "SPO nBlocks=" << e.nBlocks << " firstBlock=" << e.firstBlock << " lastBlock=" << e.lastBlock
      << " nSplines=" << e.nSplines << " nSplinesPerBlock=" << e.nSplinesPerBlock << '\n';
}

}
#include "QMCWaveFunctions/EinsplineSPODeviceImpCPU.hpp"
#ifdef QMC_USE_KOKKOS
#include "QMCWaveFunctions/EinsplineSPODeviceImpKOKKOS.hpp"
#endif
#ifdef QMC_USE_CUDA
#include "QMCWaveFunctions/EinsplineSPODeviceImpCUDA.hpp"
#endif
namespace qmcplusplus
{
#ifdef QMC_USE_KOKKOS
  //extern template class EinsplineSPODeviceImp<Devices::KOKKOS, float>;
  //extern template class EinsplineSPODeviceImp<Devices::KOKKOS, double>;
#endif
}


#endif
