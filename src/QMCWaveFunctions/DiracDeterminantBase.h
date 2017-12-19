////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by:
//
// File created by: Jeongnim Kim, jeongnim.kim@intel.com, Intel Corp.
////////////////////////////////////////////////////////////////////////////////
// -*- C++ -*-

/**
 * @file FakeWaveFunction.h
 * @brief Top level wavefunction container
 *
 * Represents a product of wavefunction components (classes based on
 * WaveFunctionComponentBase).
 *
 * Corresponds to QMCWaveFunction/TrialWaveFunction.h in the QMCPACK source.
 */

#ifndef QMCPLUSPLUS_DIRACDETERMINANTBASE_H
#define QMCPLUSPLUS_DIRACDETERMINANTBASE_H

namespace qmcplusplus
{
/** A minimal TrialWavefunction
 */
struct DiracDeterminantBase
{
  virtual void recompute()             = 0;
  virtual double ratio(int)            = 0;
  virtual void acceptMove(int)         = 0;
  virtual double operator()(int) const = 0;
  virtual int size() const             = 0;
};
} // qmcplusplus
#endif
