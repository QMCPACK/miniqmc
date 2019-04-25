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
 * @file Mover_ref.h
 * @brief Declaration of Mover class
 *
 */

#ifndef QMCPLUSPLUS_MOVER_REF_HPP
#define QMCPLUSPLUS_MOVER_REF_HPP

#include <Utilities/Configuration.h>
#include <Utilities/RandomGenerator.h>
#include <Particle/ParticleSet.h>
#include "QMCWaveFunctions/SPOSet.h"
#include <QMCWaveFunctions/WaveFunction.h>
#include <Particle/ParticleSet_builder.hpp>
#include <Input/pseudo.hpp>

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
struct Mover_ref
{
  using RealType = QMCTraits::RealType;

  /// random number generator
  RandomGenerator<RealType> rng;
  /// electrons
  ParticleSet els;
  /// single particle orbitals
  SPOSet* spo;
  /// wavefunction container
  WaveFunction wavefunction;
  /// non-local pseudo-potentials
  NonLocalPP<RealType> nlpp;

  /// constructor
  Mover_ref(const uint32_t myPrime, const ParticleSet& ions) : spo(nullptr), rng(myPrime), nlpp(rng)
  {
    build_els(els, ions, rng);
  }

  /// destructor
  ~Mover_ref()
  {
    if (spo != nullptr)
      delete spo;
  }
};

template<class T, typename TBOOL>
std::vector<T*>
    filtered_list_ref(const std::vector<T*>& input_list, const std::vector<TBOOL>& chosen)
{
  std::vector<T*> final_list;
  for (int iw = 0; iw < input_list.size(); iw++)
    if (chosen[iw])
      final_list.push_back(input_list[iw]);
  return final_list;
}

std::vector<ParticleSet*> extract_els_list_ref(const std::vector<Mover_ref*>& mover_list)
{
  std::vector<ParticleSet*> els_list;
  for (auto it = mover_list.begin(); it != mover_list.end(); it++)
    els_list.push_back(&(*it)->els);
  return els_list;
}

std::vector<SPOSet*> extract_spo_list_ref(const std::vector<Mover_ref*>& mover_list)
{
  std::vector<SPOSet*> spo_list;
  for (auto it = mover_list.begin(); it != mover_list.end(); it++)
    spo_list.push_back((*it)->spo);
  return spo_list;
}

std::vector<WaveFunction*> extract_wf_list_ref(const std::vector<Mover_ref*>& mover_list)
{
  std::vector<WaveFunction*> wf_list;
  for (auto it = mover_list.begin(); it != mover_list.end(); it++)
    wf_list.push_back(&(*it)->wavefunction);
  return wf_list;
}

} // namespace qmcplusplus

#endif
