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
// -*- C++ -*-

/**
 * @file
 * @brief Explicit instantiation of device implementation
 */

#include "Devices.h"
#include "QMCWaveFunctions/EinsplineSPODeviceImpKOKKOS.hpp"

namespace qmcplusplus
{
  template class EinsplineSPODeviceImp<Devices::KOKKOS, double>;
  template class EinsplineSPODeviceImp<Devices::KOKKOS, float>;
}
