//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2021 QMCPACK developers.
//
// File developed by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//
// File created by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//////////////////////////////////////////////////////////////////////////////////////

#include "catch.hpp"
#include "OMPTarget/OMPallocator.hpp"

namespace qmcplusplus
{

TEST_CASE("OMPallocator offset", "[openmp]")
{
  OMPallocator<int> myalloc;
  auto* host_ptr = myalloc.allocate(10);
  auto* dev_ptr = getOffloadDevicePtr(host_ptr);
  
  CHECK(getOffloadDevicePtr(host_ptr + 1) == dev_ptr + 1);
}
}
