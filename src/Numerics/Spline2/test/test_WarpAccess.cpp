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
#include "Numerics/Einspline/test/TestKernel.h"

namespace qmcplusplus
{

TEST_CASE("Test kernel launch","[CUDA]")
{
  launch_test_kernel(1);
}

}

