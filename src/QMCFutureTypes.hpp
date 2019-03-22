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
 */
#include <array>
#include "Numerics/Containers.h"

#ifndef QMCPLUSPLUS_QMC_FUTURE_TYPES_HPP
#define QMCPLUSPLUS_QMC_FUTURE_TYPES_HPP

namespace qmcplusplus
{
template <typename T>
struct QMCFutureTypes
{
  using FuturePos          = std::array<T,3>;
  using vContainer_type    = aligned_vector<T>;
  using gContainer_type    = VectorSoAContainer<T, 3>;
  using lContainer_type    = VectorSoAContainer<T, 4>;
  using hContainer_type    = VectorSoAContainer<T, 6>;
};
}
#endif
