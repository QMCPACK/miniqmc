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

#include "catch.hpp"
#include "CUDA/GPUArray.h"
#include "Numerics/Spline2/MultiBsplineCUDA.hpp"
namespace qmcplusplus
{

TEST_CASE("MultiBspline<CUDA> instantiation", "[CUDA][Spline2]")
{
  MultiBspline<Devices::CUDA, double> mbcd;
  MultiBspline<Devices::CUDA, float> mbcf;
  
}


}
