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

// Base class, and serial (no mpi) implementation
class Communicate
{
public:
  Communicate(int argc, char **argv);

  virtual ~Communicate();

  virtual void initialize(int argc, char **argv);

  int rank() { return m_rank; }
  int size() { return m_size; }
  bool root() { return m_root; }
protected:
  int m_rank;
  int m_size;
  bool m_root;
};

#ifdef HAVE_MPI

class CommunicateMPI : public Communicate
{
public:
  CommunicateMPI(int argc, char **argv);

  void initialize(int argc, char **argv) override;

  MPI_Comm world() { return m_world; }

  virtual ~CommunicateMPI();
protected:
  MPI_Comm m_world;

};
#endif

#endif
