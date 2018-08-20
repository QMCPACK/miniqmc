//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2018 Jeongnim Kim and QMCPACK developers.
//
// File developed by:  Mark Dewing, mdewing@anl.gov Argonne National Laboratory
//
// File created by: Mark Dewing, mdewing@anl.gov Argonne National Laboratory
//////////////////////////////////////////////////////////////////////////////////////


/** @file Communicate.h
 * @brief Declaration of Communicate and CommunicateMPI classes.
 */
#ifndef COMMUNICATE_H
#define COMMUNICATE_H

#include <Utilities/Configuration.h>

#ifdef HAVE_MPI
#include <mpi.h>
#endif

class Communicate
{
public:
  Communicate(int argc, char** argv);

  virtual ~Communicate();

  int rank() { return m_rank; }
  int size() { return m_size; }
  bool root() { return m_rank == 0; }
#ifdef HAVE_MPI
  MPI_Comm world() { return m_world; }
#endif
  void reduce(int& value);

protected:
  int m_rank;
  int m_size;
#ifdef HAVE_MPI
  MPI_Comm m_world;
#endif
};

#endif
