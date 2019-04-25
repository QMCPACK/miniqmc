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
#include <QMCWaveFunctions/WaveFunction.h>
#include <Particle/ParticleSet_builder.hpp>
#include <Input/pseudo.hpp>
#include "QMCWaveFunctions/einspline_spo.hpp"

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


/**
 * breaking Ye's design a bit here.  Introducing spo_psi, spo_grad and spo_hess here,
 * whereas they would probably better live in a walker.  
 */
  
template<typename T, int blockSize>
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

  ///
  using vContainer_type = typename einspline_spo<T,blockSize>::vContainer_type;
  using gContainer_type = typename einspline_spo<T,blockSize>::gContainer_type;
  using hContainer_type = typename einspline_spo<T,blockSize>::hContainer_type;

  vContainer_type spo_psi;
  gContainer_type spo_grad;
  hContainer_type spo_hess;
  
  /// constructor
  template<typename spotype>
  Mover(const uint32_t myPrime, const ParticleSet& ions, const spotype& spo) : rng(myPrime), nlpp(rng)
  {
    build_els(els, ions, rng);
    spo_psi = vContainer_type("spo_psi", spo.nSplines);
    spo_grad = gContainer_type("spo_grad", spo.nSplines);
    spo_hess = hContainer_type("spo_hess", spo.nSplines);
  }

  /// destructor
  ~Mover() = default;
};

template<class T, typename TBOOL>
std::vector<T*> filtered_list(const std::vector<T*>& input_list, const std::vector<TBOOL>& chosen)
{
  std::vector<T*> final_list;
  for (int iw = 0; iw < input_list.size(); iw++)
    if (chosen[iw])
      final_list.push_back(input_list[iw]);
  return final_list;
}

template<typename moverType>
std::vector<ParticleSet*> extract_els_list(const std::vector<moverType*>& mover_list)
{
  std::vector<ParticleSet*> els_list;
  for (auto it = mover_list.begin(); it != mover_list.end(); it++)
    els_list.push_back(&(*it)->els);
  return els_list;
}

template<typename moverType>
std::vector<typename moverType::vContainer_type> extract_spo_psi_list(const std::vector<moverType*>& mover_list)
{
  std::vector<typename moverType::vContainer_type> vals;
  for (auto it = mover_list.begin(); it != mover_list.end(); it++)
    vals.push_back((*it)->spo_psi);
  return vals;
}

template<typename moverType>
std::vector<typename moverType::gContainer_type> extract_spo_grad_list(const std::vector<moverType*>& mover_list)
{
  std::vector<typename moverType::gContainer_type> grads;
  for (auto it = mover_list.begin(); it != mover_list.end(); it++)
    grads.push_back((*it)->spo_grad);
  return grads;
}

template<typename moverType>
std::vector<typename moverType::hContainer_type> extract_spo_hess_list(const std::vector<moverType*>& mover_list)
{
  std::vector<typename moverType::hContainer_type> hesss;
  for (auto it = mover_list.begin(); it != mover_list.end(); it++)
    hesss.push_back((*it)->spo_hess);
  return hesss;
}
  
template<typename moverType>
std::vector<WaveFunction*> extract_wf_list(const std::vector<moverType*>& mover_list)
{
  std::vector<WaveFunction*> wf_list;
  for (auto it = mover_list.begin(); it != mover_list.end(); it++)
    wf_list.push_back(&(*it)->wavefunction);
  return wf_list;
}

} // namespace qmcplusplus

#endif
