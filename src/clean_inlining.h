////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2018 QMCPACK developers.
//
// File developed by:
// Peter Doak, doakpw@ornl.gov, Oak Ridge National Lab
//
// File created by:
// Peter Doak, doakpw@ornl.gov, Oak Ridge National Lab
////////////////////////////////////////////////////////////////////////////////
// -*- C++ -*-

#ifndef QMCPLUSPLUS_CLEAN_INLINING_H
#define QMCPLUSPLUS_CLEAN_INLINING_H
#include "config.h"
#ifdef QMC_USE_KOKKOS
#include <Kokkos_Core.hpp>
#else
#define KOKKOS_INLINE_FUNCTION inline
#endif

/** @file
 *  @brief Performance portability includes not requiring Kokkos
 */
#endif
