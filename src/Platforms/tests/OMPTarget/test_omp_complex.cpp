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

namespace qmcplusplus
{

template<typename T>
void test_map()
{
  std::cout << "map(complex<>)" << std::endl;
  std::complex<T> a(0.2, 1), a_check;
  #pragma omp target map(from:a_check)
  {
    a_check = a;
  }

  CHECK(a_check == ComplexApprox(a));
}

template<typename RT, typename AT, typename BT>
void test_plus(AT a, BT b)
{
  std::cout << "operator +" << std::endl;
  std::complex<RT> c, c_host;

  c_host = a + b;
  #pragma omp target map(from:c)
  {
    c = a + b;
  }

  CHECK(c == ComplexApprox(c_host));
}

template<typename RT, typename AT, typename BT>
void test_minus(AT a, BT b)
{
  std::cout << "operator -" << std::endl;
  std::complex<RT> c, c_host;

  c_host = a - b;
  #pragma omp target map(from:c)
  {
    c = a - b;
  }

  CHECK(c == ComplexApprox(c_host));
}

template<typename RT, typename AT, typename BT>
void test_mul(AT a, BT b)
{
  std::cout << "operator *" << std::endl;
  std::complex<RT> c, c_host;

  c_host = a * b;
  #pragma omp target map(from:c)
  {
    c = a * b;
  }

  CHECK(c == ComplexApprox(c_host));
}

template<typename RT, typename AT, typename BT>
void test_div(AT a, BT b)
{
  std::cout << "operator /" << std::endl;
  std::complex<RT> c, c_host;

  c_host = a / b;
  #pragma omp target map(from:c)
  {
    c = a / b;
  }

  CHECK(c == ComplexApprox(c_host));
}

template<typename T>
void test_complex()
{
  test_map<T>();

  test_plus<T>(std::complex<T>(0, 1), std::complex<T>(0.5, 0.3));
  test_plus<T>(std::complex<T>(0, 1), T(0.5));
  test_plus<T>(T(0.5), std::complex<T>(0, 1));

  test_minus<T>(std::complex<T>(0, 1), std::complex<T>(0.5, 0.3));
  test_minus<T>(std::complex<T>(0, 1), T(0.5));
  test_minus<T>(T(0.5), std::complex<T>(0, 1));

  test_mul<T>(std::complex<T>(0, 1), std::complex<T>(0.5, 0.3));
  test_mul<T>(std::complex<T>(0, 1), T(0.5));
  test_mul<T>(T(0.5), std::complex<T>(0, 1));

  test_div<T>(std::complex<T>(0, 1), std::complex<T>(0.5, 0.3));
  test_div<T>(std::complex<T>(0, 1), T(0.5));
  test_div<T>(T(0.5), std::complex<T>(0, 1));
}

TEST_CASE("complex arithmetic", "[openmp]")
{
  std::cout << "Testing float" << std::endl;
  test_complex<float>();
  std::cout << "Testing double" << std::endl;
  test_complex<double>();
}

} // namespace qmcplusplus
