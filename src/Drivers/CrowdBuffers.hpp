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
/** @file
 *  Crowd scope buffers that may be specialized for device
 *  They are assumed to be RAII with Crowd.
 */

#ifndef QMCPLUSPLUS_CROWD_BUFFERS_HPP
#define QMCPLUSPLUS_CROWD_BUFFERS_HPP
 
namespace qmcplusplus
{
/** By default there are no buffers shared at this level
 */
template<Devices DT>
class CrowdBuffers
{

};

#ifdef QMC_USE_CUDA
#include "CUDA/GPUArray.h"
#include "CUDA/PinnedHostBuffer.hpp"

/** To reduce data movement and synchronization many buffers are at the crowd scope
 *  since this is small this is not in its own header.
 */
template<>
class CrowdBuffers<Devices::CUDA>
{
public:
  using T = double;
  PinnedHostBuffer psi;
  PinnedHostBuffer grad;
  PinnedHostBuffer hess;
  GPUArray<T, 1, 1> dev_v_nlpp;
  GPUArray<T, 1, 1> dev_psi;
  GPUArray<T, 3, 1> dev_grad;
  GPUArray<T, 6, 1> dev_hess;
  MultiBsplineFuncs<Devices::CUDA, T> compute_engine;
};
#endif
}

#endif
