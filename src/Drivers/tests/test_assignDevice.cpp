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
#include <Utilities/Communicate.h>
#include <DeviceManager.h>

namespace qmcplusplus
{

TEST_CASE("assignDevice", "[openmp]")
{
  Communicate comm(0, nullptr);

  DeviceManager dm(comm.rank(), comm.size());

  if (comm.rank() == 0)
    std::cout << "number of ranks : " << comm.size() << ", number of accelerators : " << dm.getNumDevices() << std::endl;

  comm.barrier();
  std::cout << "rank id : " << comm.rank() << ", accelerator id : " << dm.getNumDevices() << std::endl;
  comm.barrier();
}

} // namespace qmcplusplus
