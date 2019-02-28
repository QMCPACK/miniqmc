////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2019 QMCPACK developers.
//
// File developed by:
// Peter Doak, doakpw@ornl.gov, Oak Ridge National Lab
//
// File created by:
// Peter Doak, doakpw@ornl.gov, Oak Ridge National Lab
////////////////////////////////////////////////////////////////////////////////

#include "catch.hpp"
#include "CUDA/GPUArray.h"


namespace qmcpluplus
{
TEST_CASE("GPUArray Instantiate", "[CUDA]") { qmcplusplus::GPUArray<double, 1> gD; }

TEST_CASE("GPUArray Resize", "[CUDA]")
{
  qmcplusplus::GPUArray<double, 1> gD;
  gD.resize(16, 10);
  // Width and Height are in bytes
  REQUIRE(gD.getWidth() == 16 * sizeof(double));
  REQUIRE(gD.getHeight() == 10 * sizeof(double));
  // Actual width of each row, this is what cuda thinks the device will work best with
  REQUIRE(gD.getPitch() >= 16 * sizeof(double));
  gD.resize(15, 10);
  REQUIRE(gD.getWidth() == 15 * sizeof(double));
  REQUIRE(gD.getHeight() == 10 * sizeof(double));
  REQUIRE(gD.getPitch() >= 16 * sizeof(double));
}

TEST_CASE("GPUArray Access", "[CUDA]")
{
  qmcplusplus::GPUArray<double, 1> gD;
  gD.resize(16, 10);
  // Width and Height are in bytes
  REQUIRE(gD.getWidth() == 16 * sizeof(double));
  REQUIRE(gD.getHeight() == 10 * sizeof(double));
  // Actual width of each row, this is what cuda thinks the device will work best with
  REQUIRE(gD.getPitch() >= 16 * sizeof(double));
  gD.resize(15, 10);
  REQUIRE(gD.getWidth() == 15 * sizeof(double));
  REQUIRE(gD.getHeight() == 10 * sizeof(double));
  REQUIRE(gD.getPitch() >= 16 * sizeof(double));
}

TEST_CASE("GPUArray Zero", "[CUDA]")
{
  qmcplusplus::GPUArray<double, 1> gD;
  gD.resize(16, 10);
  gD.zero();
}

} // namespace qmcpluplus
