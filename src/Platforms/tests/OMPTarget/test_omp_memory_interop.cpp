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
#include <iostream>
#include <stdexcept>
#include <omp.h>
#include "config.h"
#include "PinnedAllocator.h"
#include "OMPTarget/OMPallocator.hpp"

namespace qmcplusplus
{

const int array_size = 100;

template<class T, class ALLOC>
void test_pinned_memory_omp_access()
{
  std::vector<T, OMPallocator<T, ALLOC>> array(array_size, 1);
  int* array_ptr = array.data();

  #pragma omp target teams distribute parallel for map(always, tofrom: array_ptr[:array_size])
  for (int i = 0; i < array_size; i++)
  {
    array_ptr[i] += i;
  }

  int sum = 0;
  for (int i = 0; i < array_size; i++)
    sum += std::abs(i + 1 - array[i]);
  CHECK(sum == 0);
}

template<class T, class ALLOC>
void test_vendor_device_memory_omp_access()
{
  ALLOC allocator;
  int* array_ptr = allocator.allocate(array_size);

  int sum = 0;
  #pragma omp target teams distribute parallel for is_device_ptr(array_ptr) reduction(+:sum)
  for (int i = 0; i < array_size; i++)
  {
    array_ptr[i] = i;
    sum += array_ptr[i];
  }

#if defined(QMC_ENABLE_CUDA)
  cudaErrorCheck(cudaMemset(array_ptr, 0, array_size * sizeof(int)), "cudaMemset failed on ALLOC memory!");
#elif defined(QMC_ENABLE_ROCM)
  hipErrorCheck(hipMemset(array_ptr, 0, array_size * sizeof(int)), "hipMemset failed on ALLOC memory!");
#endif
  CHECK(sum == (array_size - 1) * array_size / 2);
  allocator.deallocate(array_ptr, array_size);
}

TEST_CASE("memory_interop", "[openmp]")
{
  std::cout << "test memory_interop map" << std::endl;
#if defined(QMC_ENABLE_CUDA) || defined(QMC_ENABLE_ROCM) || defined(QMC_ENABLE_ONEAPI)
  test_pinned_memory_omp_access<int, PinnedAlignedAllocator<int>>();
#endif

#if defined(QMC_ENABLE_ROCM)
  test_pinned_memory_omp_access<int, HIPHostAllocator<int>>();
#endif

  std::cout << "test memory_interop vendor device alloc" << std::endl;
#if defined(QMC_ENABLE_CUDA)
  test_vendor_device_memory_omp_access<int, CUDAAllocator<int>>();
#elif defined(QMC_ENABLE_ROCM)
  test_vendor_device_memory_omp_access<int, HIPAllocator<int>>();
#endif

  std::cout << "test memory_interop omp_target_alloc" << std::endl;
  int* array = (int*)omp_target_alloc(array_size * sizeof(int), omp_get_default_device());
#if defined(QMC_ENABLE_CUDA)
  CUDAfill_n(array, array_size, 0);
  REQUIRE(isCUDAPtrDevice(array));
#elif defined(QMC_ENABLE_ROCM)
  hipErrorCheck(hipMemset(array, 0, array_size * sizeof(int)), "hipMemset failed on omp_target_alloc memory!");
  REQUIRE(isHIPPtrDevice(array));
#endif
  omp_target_free(array, omp_get_default_device());
}

} // namespace qmcplusplus
