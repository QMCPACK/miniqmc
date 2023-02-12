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
#include "Utilities/Configuration.h"
#include "Numerics/my_math.hpp"

namespace qmcplusplus
{

template<typename T>
void test_modf(T x)
{
  T dx;
  int intx;

  PRAGMA_OFFLOAD("omp target map(from: intx, dx)")
  {
    T ipart;
    dx = std::modf(x, &ipart);
    intx = static_cast<int>(ipart);
  }

  T dx_host;
  int intx_host;
  {
    T ipart;
    dx_host = std::modf(x, &ipart);
    intx_host = static_cast<int>(ipart);
  }

  CHECK(intx == intx_host);
  CHECK(dx == dx_host);
}

template<typename T>
void test_my_modf(T x)
{
  T dx;
  int intx;

  {
    T ipart;
    dx = my_modf(x, &ipart);
    intx = static_cast<int>(ipart);
  }

  T dx_host;
  int intx_host;
  {
    T ipart;
    dx_host = std::modf(x, &ipart);
    intx_host = static_cast<int>(ipart);
  }

  CHECK(intx == intx_host);
  CHECK(dx == dx_host);
}

TEST_CASE("float_point_break_up", "[openmp]")
{
  test_modf<float>(1.5);
  test_modf<float>(-0.5);
  test_modf<float>(0.0);
  test_modf<float>(1.0);
  test_modf<float>(-1.0);
  test_modf<double>(1.5);
  test_modf<double>(-0.5);
  test_modf<double>(0.0);
  test_modf<double>(1.0);
  test_modf<double>(-1.0);
}

TEST_CASE("float_point_my_break_up", "[openmp]")
{
  test_my_modf<float>(1.5);
  test_my_modf<float>(-0.5);
  test_my_modf<float>(0.0);
  test_my_modf<float>(1.0);
  test_my_modf<float>(-1.0);
  test_my_modf<double>(1.5);
  test_my_modf<double>(-0.5);
  test_my_modf<double>(0.0);
  test_my_modf<double>(1.0);
  test_my_modf<float>(-1.0);
}

template<typename T>
void test_sin_cos(T x)
{
  std::cout << "Testing sin and cos" << std::endl;
  T sin_v, cos_v;

  PRAGMA_OFFLOAD("omp target map(from: sin_v, cos_v)")
  {
    sin_v = std::sin(x);
    cos_v = std::cos(x);
  }

  T sin_v_host, cos_v_host;
  {
    sin_v_host = std::sin(x);
    cos_v_host = std::cos(x);
  }

  CHECK(sin_v == ValueApprox(sin_v_host));
  CHECK(cos_v == ValueApprox(cos_v_host));
}

TEST_CASE("sin_cos", "[openmp]")
{
  std::cout << "Testing float" << std::endl;
  test_sin_cos<float>(1.5);
  test_sin_cos<float>(-0.5);
  std::cout << "Testing double" << std::endl;
  test_sin_cos<double>(1.5);
  test_sin_cos<double>(-0.5);
}

template<typename T>
void test_sincos(T x)
{
  std::cout << "Testing sincos" << std::endl;
  T sin_v, cos_v;

  PRAGMA_OFFLOAD("omp target map(from: sin_v, cos_v)")
  {
    qmcplusplus::sincos(x, &sin_v, &cos_v);
  }

  T sin_v_host, cos_v_host;
  {
    qmcplusplus::sincos(x, &sin_v_host, &cos_v_host);
  }

  CHECK(sin_v == ValueApprox(sin_v_host));
  CHECK(cos_v == ValueApprox(cos_v_host));
}

template<typename T>
void test_sincos_vector(T x, int size)
{
  std::cout << "Testing sincos vector" << std::endl;
  std::vector<T> sin_v(size), cos_v(size);

  const int team_size = 79;
  int n_teams = (size + team_size - 1) / team_size;
  T* sin_v_ptr = sin_v.data();
  T* cos_v_ptr = cos_v.data();
  PRAGMA_OFFLOAD("omp target teams distribute map(from: sin_v_ptr[:size], cos_v_ptr[:size])")
  for (int team_id = 0; team_id < n_teams; team_id++)
  {
    int first = team_size * team_id;
    int last = std::min(team_size * (team_id + 1), size);
    PRAGMA_OFFLOAD("omp parallel for simd")
    for (int member_id = first; member_id < last; member_id++)
      qmcplusplus::sincos(x * member_id / size, sin_v_ptr + member_id, cos_v_ptr + member_id);
  }

  for (int member_id = 0; member_id < size; member_id++)
  {
    T sin_v_host, cos_v_host;
    qmcplusplus::sincos(x * member_id / size, &sin_v_host, &cos_v_host);
    CHECK(sin_v[member_id] == ValueApprox(sin_v_host));
    CHECK(cos_v[member_id] == ValueApprox(cos_v_host));
  }

}

TEST_CASE("sincos", "[openmp]")
{
  std::cout << "Testing float" << std::endl;
  test_sincos(1.5f);
  test_sincos(-0.5f);
  test_sincos_vector(0.5f, 126);
  std::cout << "Testing double" << std::endl;
  test_sincos(1.5);
  test_sincos(-0.5);
  test_sincos_vector(0.5, 126);
}

} // namespace qmcplusplus
