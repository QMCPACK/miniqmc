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


/** @file Communicate.cpp
 * @brief Defintion of Communicate and CommunicateMPI classes.
 */
#include <Utilities/Communicate.h>
#include <iostream>

Communicate::Communicate(int argc, char **argv)
{
  initialize(argc, argv);
}

void
Communicate::initialize(int argc, char **argv)
{
  m_rank = 0;
  m_rank = 1;
  m_root = true;
}

Communicate::~Communicate()
{
}

#ifdef HAVE_MPI
CommunicateMPI::CommunicateMPI(int argc, char **argv):Communicate(argc, argv)
{
  initialize(argc, argv);
}

void
CommunicateMPI::initialize(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  m_world = MPI_COMM_WORLD;
  MPI_Comm_rank(m_world, &m_rank);
  MPI_Comm_size(m_world, &m_size);
  m_root = (m_rank == 0);
}

CommunicateMPI::~CommunicateMPI()
{
  MPI_Finalize();
}
#endif
