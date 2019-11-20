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

  REQUIRE(intx == intx_host);
  REQUIRE(dx == dx_host);
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

  REQUIRE(intx == intx_host);
  REQUIRE(dx == dx_host);
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

  REQUIRE(sin_v == ValueApprox(sin_v_host));
  REQUIRE(cos_v == ValueApprox(cos_v_host));
}

TEST_CASE("sin_cos", "[openmp]")
{
  test_sin_cos<float>(1.5);
  test_sin_cos<float>(-0.5);
  test_sin_cos<double>(1.5);
  test_sin_cos<double>(-0.5);
}

template<typename T>
void test_sincos(T x);

template<>
void test_sincos<float>(float x)
{
  float sin_v, cos_v;

  PRAGMA_OFFLOAD("omp target map(from: sin_v, cos_v)")
  {
    sincosf(x, &sin_v, &cos_v);
  }

  float sin_v_host, cos_v_host;
  {
    sincosf(x, &sin_v_host, &cos_v_host);
  }

  REQUIRE(sin_v == ValueApprox(sin_v_host));
  REQUIRE(cos_v == ValueApprox(cos_v_host));
}

template<>
void test_sincos<double>(double x)
{
  double sin_v, cos_v;

  PRAGMA_OFFLOAD("omp target map(from: sin_v, cos_v)")
  {
    sincos(x, &sin_v, &cos_v);
  }

  double sin_v_host, cos_v_host;
  {
    sincos(x, &sin_v_host, &cos_v_host);
  }

  REQUIRE(sin_v == ValueApprox(sin_v_host));
  REQUIRE(cos_v == ValueApprox(cos_v_host));
}

TEST_CASE("sincos", "[openmp]")
{
  test_sincos(1.5f);
  test_sincos(-0.5f);
  test_sincos(1.5);
  test_sincos(-0.5);
}

} // namespace qmcplusplus
