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
#include <complex>
#include <iostream>

namespace qmcplusplus
{

TEST_CASE("sqrt", "[openmp]")
{
  CHECK( Approx(1.414213562) == std::sqrt(2.0));
  CHECK( Approx(1.414213562) == std::sqrt(2.0f));
  CHECK( Approx(1.414213562) == std::sqrt(2));
}

} // namespace qmcplusplus
