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
 * @brief CPU implementation of EinsplineSPO
 */

#include <memory>
#include "Devices.h"
#include "Numerics/Spline2/BsplineSet.hpp"
#include "Utilities/SIMD/allocator.hpp"
#include "QMCWaveFunctions/EinsplineSPODeviceImpCPU.hpp"

namespace qmcplusplus
{
  template class EinsplineSPODeviceImp<Devices::CPU, float>;
  template class EinsplineSPODeviceImp<Devices::CPU, double>;
}

//This need to get the device number via hana or something stable
template class std::vector<multi_UBspline_3d_d<qmcplusplus::Devices::CPU>*, qmcplusplus::Mallocator<multi_UBspline_3d_d<(qmcplusplus::Devices)0>*, 32ul> >;
template class std::vector<qmcplusplus::VectorSoAContainer<double, 6u>, qmcplusplus::Mallocator<qmcplusplus::VectorSoAContainer<double, 6u>, 32ul> >;
template class std::vector<std::vector<double, qmcplusplus::Mallocator<double, 32ul>>, qmcplusplus::Mallocator<std::vector<double, qmcplusplus::Mallocator<double, 32ul>>,32ul>>;
template class std::vector<double, qmcplusplus::Mallocator<double, 32ul>>;

