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

#ifndef QMCPLUSPLUS_CROWD_CUDA_HPP
#define QMCPLUSPLUS_CROWD_CUDA_HPP

#include <memory>
#include <functional>
#include <type_traits>
#include <stdexcept>
#include <boost/tuple/tuple.hpp>
#include <boost/iterator/zip_iterator.hpp>

#include "Devices.h"
#include "QMCWaveFunctions/WaveFunction.h"
#include "QMCWaveFunctions/SPOSet.h"
#include "Input/pseudo.hpp"
#include "Utilities/RandomGenerator.h"
#include "Utilities/PrimeNumberSet.h"
#include "Drivers/Crowd.hpp"

namespace qmcplusplus
{
    template<Devices DT>
    class Crowd;
    
extern template class Crowd<Devices::CUDA>;

} // namespace qmcplusplus

#endif
