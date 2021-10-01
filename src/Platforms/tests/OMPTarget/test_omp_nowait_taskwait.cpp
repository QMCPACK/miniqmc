//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2019 QMCPACK developers.
//
// File developed by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//
// File created by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//////////////////////////////////////////////////////////////////////////////////////


#include "catch.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include "PinnedAllocator.h"
#include "OMPTarget/OMPallocator.hpp"

namespace qmcplusplus
{

const int num_sections = 1;
const int section_size = 100;
constexpr int array_size = num_sections * section_size;

TEST_CASE("nowait_taskwait", "[openmp]")
{
  //std::vector<int, OMPallocator<int>> array(array_size, 1);
  std::vector<int, OMPallocator<int, PinnedAlignedAllocator<int>>> array(array_size, 1);
  int* array_ptr = array.data();

  #pragma omp target update to(array_ptr[:array_size])
  #pragma omp target teams distribute parallel for map(tofrom: array_ptr[:array_size])
  for (int i = 0; i < array_size; i++)
  {
    array_ptr[i] += i;
  }

  CHECK(array_ptr[4] == 1);
  CHECK(array_ptr[94] == 1);
  for (int offset = 0; offset < array_size; offset += section_size)
  {
    #pragma omp target update from(array_ptr[offset:section_size]) nowait
  }
  #pragma omp taskwait
  CHECK(array_ptr[4] == 5);
  CHECK(array_ptr[94] == 95);
}

} // namespace qmcplusplus
