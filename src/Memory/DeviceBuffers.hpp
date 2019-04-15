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
 *  Devices require all sorts of buffers, this is a pretty generic container to hold them.
 *  Specialized versions of the class template can contain whatever specialized methods useing this need
 */

#ifndef QMCPLUSPLUS_DEVICE_BUFFERS_HPP
#define QMCPLUSPLUS_DEVICE_BUFFERS_HPP

#include "Devices.h"

namespace qmcplusplus
{
template <Devices DT>
struct DeviceBuffers
{    
};

#ifdef QMC_USE_CUDA
#include "CUDA/PinnedHostBuffer.hpp"

/** This is used to share one pinned host buffer between all the dterminants in a crowd
 *  having many in play at once seems to cause performance issues
 */
template<>
struct DeviceBuffers<Devices::CUDA>
{
  PinnedHostBuffer determinant_host_buffer;
};
#endif
}

#endif
