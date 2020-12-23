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
#include "Utilities/Configuration.h"

namespace qmcplusplus
{
template<typename T>
void test_icpx_opencl_wrong_number()
{
  T q[1];
  T* y_host[1];
  T* q_ptr  = q;
  T** y_ptr = y_host;
#pragma omp target enter data map(alloc : q_ptr[:1])
#pragma omp target data use_device_ptr(q_ptr)
  {
    y_host[0] = q_ptr;
  }


#pragma omp target data map(y_ptr[:1]) use_device_ptr(y_ptr)
  {
    T** y        = reinterpret_cast<T**>(y_ptr);
    T foo_before = 2;
    T foo_after;
#pragma omp target is_device_ptr(y) map(tofrom : foo_before)
    {
      y[0][0] = foo_before;
      printf("y[0][0] before %p %lf\n", y[0], y[0][0]);
    }

#pragma omp target is_device_ptr(y) map(tofrom : foo_after)
    {
      printf("y[0][0] after %p %lf\n", y[0], y[0][0]);
      foo_after = y[0][0];
    }

    printf("foo before %lf\n", foo_before);
    printf("foo after %lf\n", foo_after);
    REQUIRE(std::abs(foo_before - foo_after) < 1E-9);
  }
}

TEST_CASE("icpx OpenCL wrong number", "[openmp]")
{
  std::cout << "testing float" << std::endl;
  test_icpx_opencl_wrong_number<float>();
  std::cout << "testing double" << std::endl;
  test_icpx_opencl_wrong_number<double>();
}

} // namespace qmcplusplus
