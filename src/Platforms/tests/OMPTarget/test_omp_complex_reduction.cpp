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
#include <complex>
#include "Utilities/Configuration.h"
#include "Numerics/my_math.hpp"
#include "OMPTarget/ompReduction.hpp"

namespace qmcplusplus
{

template<typename T>
void test_map()
{
  std::cout << "map(complex<>)" << std::endl;
  std::complex<T> a(0.2, 1), a_check;
#pragma omp target map(from : a_check)
  {
    a_check = a;
  }

  CHECK(a_check == ComplexApprox(a));
}

template<typename T>
void test_reduction()
{
  std::cout << "flat parallelism" << std::endl;
  std::complex<T> sum(0), sum_host(0);
  const int size = 100;
  std::complex<T> array[size];
  for (int i = 0; i < size; i++)
  {
    array[i] = {T(i), T(-i)};
    sum_host += array[i];
  }

#pragma omp target teams distribute parallel for map(to : array[:size]) reduction(+ : sum)
  for (int i = 0; i < size; i++)
    sum += array[i];

  CHECK(sum == ComplexApprox(sum_host));

  std::cout << "hierarchical parallelism" << std::endl;
  const int nblock(10), block_size(10);
  std::complex<T> block_sum[nblock];
#pragma omp target teams distribute map(to : array[:size]) map(from : block_sum[:nblock])
  for (int ib = 0; ib < nblock; ib++)
  {
    std::complex<T> partial_sum;
    const int istart = ib * block_size;
    const int iend   = (ib + 1) * block_size;
#pragma omp parallel for reduction(+ : partial_sum)
    for (int i = istart; i < iend; i++)
      partial_sum += array[i];
    block_sum[ib] = partial_sum;
  }

  sum = 0;
  for (int ib = 0; ib < nblock; ib++)
    sum += block_sum[ib];

  CHECK(sum == ComplexApprox(sum_host));
}

template<typename T>
void test_complex()
{
  test_map<T>();
  test_reduction<T>();
}

TEST_CASE("complex reduction", "[openmp]")
{
  std::cout << "Testing float" << std::endl;
  test_complex<float>();
  std::cout << "Testing double" << std::endl;
  test_complex<double>();
}

} // namespace qmcplusplus
