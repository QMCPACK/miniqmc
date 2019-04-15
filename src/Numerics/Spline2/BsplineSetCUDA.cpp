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

#include "BsplineSet.hpp"

namespace qmcplusplus
{
template class BsplineSet<Devices::CUDA, double>;
template class BsplineSet<Devices::CUDA, float>;
template class BsplineSetCreator<Devices::CUDA, double>;
template class BsplineSetCreator<Devices::CUDA, float>;

} // namespace qmcplusplus