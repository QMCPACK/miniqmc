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
#include "OMPTarget/OMPallocator.hpp"

namespace qmcplusplus
{

const int array_size = 100;

TEST_CASE("partial_update", "[openmp]")
{
  std::vector<int, OMPallocator<int>> array(array_size, 1);
  int* array_ptr = array.data();

  #pragma omp target teams distribute parallel for
  for (int i = 0; i < array_size; i++)
  {
    array_ptr[i] += i;
  }

  const int offset = 4;
  REQUIRE(array_ptr[offset] == 1);
  #pragma omp target update from(array_ptr[offset:(array_size - offset)])
  REQUIRE(array_ptr[offset] == offset + 1);
}

} // namespace qmcplusplus
