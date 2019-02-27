////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2019 QMCPACK developers.
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
 * @brief CUDA implementation of EinsplineSPO
 * This leverages the legacy CUDA einspline code and has some design quirks due to that
 */

#include "Devices.h"
#include "QMCWaveFunctions/EinsplineSPODeviceImpCUDA.hpp"

namespace qmcplusplus
{
  template class EinsplineSPODeviceImp<Devices::CUDA, float>;
  template class EinsplineSPODeviceImp<Devices::CUDA, double>;
}

//template class std::vector<multi_UBspline_3d_d<qmcplusplus::Devices::CUDA>*, qmcplusplus::Mallocator<multi_UBspline_3d_d<(qmcplusplus::Devices)1>*, 32ul> >;
// template class std::vector<qmcplusplus::VectorSoAContainer<double, 6u>, qmcplusplus::Mallocator<qmcplusplus::VectorSoAContainer<double, 6u>, 32ul> >;
// template class std::vector<std::vector<double, qmcplusplus::Mallocator<double, 32ul>>, qmcplusplus::Mallocator<std::vector<double, qmcplusplus::Mallocator<double, 32ul>>,32ul>>;
// template class std::vector<double, qmcplusplus::Mallocator<double, 32ul>>;

