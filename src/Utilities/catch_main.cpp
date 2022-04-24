//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by: Mark Dewing, markdewing@gmail.com, University of Illinois at Urbana-Champaign
//
// File created by: Mark Dewing, markdewing@gmail.com, University of Illinois at Urbana-Champaign
//////////////////////////////////////////////////////////////////////////////////////


#define CATCH_CONFIG_RUNNER
#include "catch.hpp"
#include "DeviceManager.h"

#ifdef CATCH_MAIN_HAVE_MPI
#include "Communicate.h"
#endif

// Replacement unit test main function to ensure that MPI is finalized once
// (and only once) at the end of the unit test.

int main(int argc, char* argv[])
{
  Catch::Session session;
  // Parse arguments.
  int parser_err = session.applyCommandLine(argc, argv);
#ifdef CATCH_MAIN_HAVE_MPI
  Communicate comm(argc, argv);
  // assign accelerators within a node
  qmcplusplus::DeviceManager::initializeGlobalDeviceManager(comm.rank(), comm.size());
#else
  qmcplusplus::DeviceManager::initializeGlobalDeviceManager(0, 1);
#endif
  // Run the tests.
  int result = session.run(argc, argv);
  if (parser_err != 0)
  {
    return parser_err;
  }
  else
  {
    return result;
  }
}
