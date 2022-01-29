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

TEST_CASE("task depend taskwait", "[openmp]")
{
  int a = 0;
  #pragma omp target map(tofrom: a) depend(out: a) nowait
  {
    int sum = 0;
    for (int i = 0; i < 100000; i++)
      sum++;
    a = 1;
  }

  #pragma omp task depend(in: a) shared(a)
  {
    REQUIRE(a == 1);
  }

  #pragma omp taskwait
}

} // namespace qmcplusplus
