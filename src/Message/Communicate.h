//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by: Ken Esler, kpesler@gmail.com, University of Illinois at
// Urbana-Champaign
//                    Miguel Morales, moralessilva2@llnl.gov, Lawrence Livermore
//                    National Laboratory
//                    Jeremy McMinnis, jmcminis@gmail.com, University of
//                    Illinois at Urbana-Champaign
//                    Jeongnim Kim, jeongnim.kim@gmail.com, University of
//                    Illinois at Urbana-Champaign
//                    Mark Dewing, markdewing@gmail.com, University of Illinois
//                    at Urbana-Champaign
//                    Mark A. Berrill, berrillma@ornl.gov, Oak Ridge National
//                    Laboratory
//
// File created by: Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois
// at Urbana-Champaign
//////////////////////////////////////////////////////////////////////////////////////

#ifndef OHMMS_COMMUNICATE_H
#define OHMMS_COMMUNICATE_H
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#define APP_ABORT(msg)                                            \
  {                                                               \
    std::cerr << "Fatal Error. Aborting at " << msg << std::endl; \
    exit(1);                                                      \
  }

#endif // OHMMS_COMMUNICATE_H
