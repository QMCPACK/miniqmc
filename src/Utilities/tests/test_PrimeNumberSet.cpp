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
#include "Utilities/PrimeNumberSet.h"

TEST_CASE("PrimeNumberSet Basic", "[Utilities]") {
    PrimeNumberSet<uint32_t> myPrimes;
    REQUIRE(myPrimes[0] == 3);
    REQUIRE(myPrimes[1] == 5);
    REQUIRE(myPrimes[7] == 23);
}

