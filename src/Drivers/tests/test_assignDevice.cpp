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
#include "OMPTarget/assignDevice.h"
#include <Utilities/Communicate.h>

namespace qmcplusplus
{

TEST_CASE("assignDevice", "[openmp]")
{
  Communicate comm(0, nullptr);

  int num_accelerators = 0;
  int assigned_accelerators_id = -1;

  assignDevice(num_accelerators, assigned_accelerators_id, comm.rank(), comm.size());
}

} // namespace qmcplusplus
