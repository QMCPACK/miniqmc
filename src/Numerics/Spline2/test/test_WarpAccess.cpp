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
#include "Numerics/Spline2/test/TestKernel.h"
#include "Numerics/Spline2/test/TestMultiBspline.hpp"

namespace qmcplusplus
{

TEST_CASE("WarpAccess Basic Launch","[CUDA]")
{
  launch_test_kernel(1);
}

TEST_CASE("WarpAccess With Spline","[CUDA]")
{
  TestMultiBspline<double,double> tmb(100);
  tmb.create();
  launch_test_kernel_with_spline(tmb.cuda_spline, 1);
}

  
}

