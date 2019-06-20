////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by:
// Ye Luo, yeluo@anl.gov,
//    Argonne National Laboratory
//
// File created by:
// Ye Luo, yeluo@anl.gov,
//    Argonne National Laboratory
////////////////////////////////////////////////////////////////////////////////
// -*- C++ -*-

/**
 * @file Mover.h
 * @brief Declaration of Mover class
 *
 */

#ifndef QMCPLUSPLUS_MOVER_HPP
#define QMCPLUSPLUS_MOVER_HPP

#include <Utilities/Configuration.h>
#include <Utilities/RandomGenerator.h>
#include <Particle/ParticleSet.h>
#include "QMCWaveFunctions/SPOSet.h"
#include <QMCWaveFunctions/WaveFunction.h>
#include <Particle/ParticleSet_builder.hpp>
#include <Drivers/NonLocalPP.hpp>

namespace qmcplusplus
{
/**
   * @brief Container class includes all the classes required to operate walker moves
   *
   * Movers are distinct from walkers. Walkers are lightweight in memory, while
   * Movers carry additional data needed to evolve the walkers efficiently.
   * In a memory capacity limited scenario, a limited number of movers can be used to
   * handle a large amount of walkers.
   *
   * This class is used only by QMC drivers.
   */
struct Mover
{
  using RealType = QMCTraits::RealType;

  /// random number generator
  RandomGenerator<RealType> rng;
  /// electrons
  ParticleSet els;
  /// wavefunction container
  WaveFunction wavefunction;
  /// non-local pseudo-potentials
  NonLocalPP<RealType> nlpp;

  /// constructor
  Mover(const uint32_t myPrime, const ParticleSet& ions) : rng(myPrime), nlpp(rng, ions)
  {
    build_els(els, ions, rng);
  }
};

inline void FairDivideLow(int ntot, int nparts, int me, int& first, int& last)
{
  int bat     = ntot / nparts;
  int residue = nparts - ntot % nparts;
  if(me<residue)
  {
    first = bat * me;
    last  = bat * (me + 1);
  }
  else
  {
    first = (bat + 1) * me - residue;
    last  = (bat + 1) * (me + 1) - residue;
  }
}

const std::vector<Mover*> extract_sub_list(const std::vector<Mover*>& mover_list, int first, int last)
{
  std::vector<Mover*> sub_list;
  for (auto it = mover_list.begin() + first; it != mover_list.begin() + last; it++)
    sub_list.push_back(*it);
  return sub_list;
}

const std::vector<ParticleSet*> extract_els_list(const std::vector<Mover*>& mover_list)
{
  std::vector<ParticleSet*> els_list;
  for (auto it = mover_list.begin(); it != mover_list.end(); it++)
    els_list.push_back(&(*it)->els);
  return els_list;
}

const std::vector<WaveFunction*> extract_wf_list(const std::vector<Mover*>& mover_list)
{
  std::vector<WaveFunction*> wf_list;
  for (auto it = mover_list.begin(); it != mover_list.end(); it++)
    wf_list.push_back(&(*it)->wavefunction);
  return wf_list;
}

const std::vector<NonLocalPP<QMCTraits::RealType>*> extract_nlpp_list(const std::vector<Mover*>& mover_list)
{
  std::vector<NonLocalPP<QMCTraits::RealType>*> nlpp_list;
  for (auto it = mover_list.begin(); it != mover_list.end(); it++)
    nlpp_list.push_back(&(*it)->nlpp);
  return nlpp_list;
}

} // namespace qmcplusplus

#endif
