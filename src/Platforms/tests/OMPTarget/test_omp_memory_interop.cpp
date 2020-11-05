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
#include <omp.h>
#include "Utilities/Configuration.h"
#include "Numerics/my_math.hpp"
#include "PinnedAllocator.h"

namespace qmcplusplus
{

const int array_size = 100;

template<class T, class ALLOC>
void test_device_memory_omp_access()
{
  std::vector<T, ALLOC> array(array_size, 1);
  int* array_ptr = array.data();

  #pragma omp target teams distribute parallel for is_device_ptr(array_ptr)
  for (int i = 0; i < array_size; i++)
  {
    array_ptr[i] += i;
  }

  int sum = 0;
  for (int i = 0; i < array_size; i++)
    sum += std::abs(i + 1 - array[i]);
  REQUIRE(sum == 0);
}

TEST_CASE("memory_interop", "[openmp]")
{
#if defined(QMC_ENABLE_CUDA) || defined(QMC_ENABLE_ROCM) || defined(QMC_ENABLE_ONEAPI)
  test_device_memory_omp_access<int, PinnedAlignedAllocator<int>>();
#endif

  int* array = (int*)omp_target_alloc(array_size, omp_get_default_device());
#if defined(QMC_ENABLE_CUDA)
  REQUIRE(isCUDAPtrDevice(array));
#elif defined(QMC_ENABLE_ROCM)
  REQUIRE(isHIPPtrDevice(array));
#endif
  omp_target_free(array, omp_get_default_device());
}

} // namespace qmcplusplus
