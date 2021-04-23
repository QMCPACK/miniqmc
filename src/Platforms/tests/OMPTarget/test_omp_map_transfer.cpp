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
#include <stdexcept>
#include "PinnedAllocator.h"
#include "OMPTarget/OMPallocator.hpp"

namespace qmcplusplus
{

TEST_CASE("map_always", "[openmp]")
{
  constexpr int array_size = 64 << 20;
  std::vector<int, OMPallocator<int, PinnedAlignedAllocator<int>>> array(array_size, 1);
  int* array_ptr = array.data();

  for (int i = 0; i < 2; i++)
  {
    #pragma omp target update to(array_ptr [:array_size])
    #pragma omp target update from(array_ptr [:array_size])
  }
}

} // namespace qmcplusplus
