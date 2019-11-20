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

TEST_CASE("reduction with parallel for", "[openmp]")
{
  int counts1 = 0;
  #pragma omp target teams map(from:counts1)
  {
    int counts_team = 0;
    #pragma omp parallel for
    for (int i=0; i<4; i++)
      #pragma omp atomic
      counts_team += 1;
    counts1 = counts_team;
  }
  REQUIRE(counts1 == 4);

  int counts2 = 0;
  #pragma omp target teams map(from:counts2)
  {
    int counts_team = 0;
    #pragma omp parallel for reduction(+:counts_team)
      for (int i=0; i<4; i++)
        counts_team += 1;
    counts2 = counts_team;
  }
  REQUIRE(counts2 == 4);
}

TEST_CASE("reduction with parallel and for split", "[openmp]")
{
  int counts1 = 0;
  #pragma omp target teams map(from:counts1)
  {
    int counts_team = 0;
    #pragma omp parallel
    {
      #pragma omp for
      for (int i=0; i<4; i++)
        #pragma omp atomic
        counts_team += 1;
    }
    counts1 = counts_team;
  }
  REQUIRE(counts1 == 4);

  int counts2 = 0;
  #pragma omp target teams map(from:counts2)
  {
    int counts_team = 0;
    #pragma omp parallel
    {
      #pragma omp for reduction(+:counts_team)
      for (int i=0; i<4; i++)
        counts_team += 1;
    }
    counts2 = counts_team;
  }
  REQUIRE(counts2 == 4);
}

} // namespace qmcplusplus
