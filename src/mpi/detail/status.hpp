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
#ifndef QMCPLUSPLUS_STATUS_H
#define QMCPLUSPLUS_STATUS_H
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
namespace qmcplusplus { namespace mpi {

#if defined(HAVE_MPI)
  struct status
  {
    status(){}

    inline operator MPI_Status&() { return m_status; }
    inline operator const MPI_Status&() const { return m_status; }

    mutable MPI_Status m_status;
  };
#else
  typedef int status;
#endif

}}
#endif
