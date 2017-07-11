//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by: 
//
// File created by: Jeongnim Kim, jeongnim.kim@inte.com, Intel Corp.
//////////////////////////////////////////////////////////////////////////////////////
//
#ifndef QMCPLUSPLUS_REQUEST_H
#define QMCPLUSPLUS_REQUEST_H
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
namespace qmcplusplus { namespace mpi {

  class communicator;
  class status;

  struct request
  {
    request();
    status wait();
    MPI_Request my_requests[2];
  };
#if !defined(HAVE_MPI)
  inline request::request() { }
  inline status request::wait() { return 0;}
#endif
}}
#endif
