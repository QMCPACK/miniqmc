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
#include <OMPTarget/assignDevice.h>
#include <Utilities/Communicate.h>

namespace qmcplusplus
{

TEST_CASE("assignDevice", "[openmp]")
{
  Communicate comm(0, nullptr);

  int num_accelerators = 0;
  int assigned_accelerators_id = -1;

  assignDevice(num_accelerators, assigned_accelerators_id, comm.rank(), comm.size());

  if (comm.rank() == 0)
    std::cout << "number of ranks : " << comm.size() << ", number of accelerators : " << num_accelerators << std::endl;
  comm.barrier();
  std::cout << "rank id : " << comm.rank() << ", accelerator id : " << assigned_accelerators_id << std::endl;
  comm.barrier();
}

} // namespace qmcplusplus
