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

#ifndef QMCPLUSPLUS_DEVICES_H
#define QMCPLUSPLUS_DEVICES_H

namespace qmcplusplus
{

/** Enum of Devices supported by this build
 *  used as a template parameter for device specializations
 *  and for template metaprogramming.
 *  I think it should be possible to not have to have duplicate
 *  Enum and names code.
 */
enum class Devices
{
  CPU,
#ifdef QMC_USE_KOKKOS
  KOKKOS,
#endif
#ifdef QMC_USE_OMPOL
  OMPOL,
#endif
#ifdef QMC_USE_CUDA
  CUDA,
#endif
  LAST // this is the end
};


} // namespace qmcplusplus

#endif
