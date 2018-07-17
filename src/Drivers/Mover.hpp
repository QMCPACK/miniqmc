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
#include <QMCWaveFunctions/einspline_spo.hpp>
#include <Numerics/Spline2/MultiBspline.hpp>
#include <QMCWaveFunctions/WaveFunction.h>
#include <Particle/ParticleSet_builder.hpp>
#include <Input/pseudo.hpp>

namespace qmcplusplus
{

  /**
   * @brief Container class includes all the classes required to operate walker moves
   *
   * Movers are different from walkers. The former carry all the data needed to evolve walkers.
   * The latter is lightweight in memory. In a capacity limited scenario,
   * a limited amount movers can be used to handle a large amount of walkers.
   *
   * This class is used only by QMC drivers.
   */
  struct Mover
  {
    using RealType = QMCTraits::RealType;
    using spo_type = einspline_spo<RealType, MultiBspline<RealType> >;

    /// random number generator
    RandomGenerator<RealType> rng;
    /// electrons
    ParticleSet               els;
    /// single particle orbitals
    spo_type                  spo;
    /// wavefunction container
    WaveFunction              wavefunction;
    /// non-local pseudo-potentials
    NonLocalPP<RealType>      nlpp;

    /// constructor
    Mover(const spo_type &spo_main,
          const int team_size,
          const int member_id,
          const uint32_t myPrime,
          const ParticleSet &ions)
      : spo(spo_main, team_size, member_id), rng(myPrime), nlpp(rng)
    {
      build_els(els, ions, rng);
    }

  };

  template <class T>
  const std::vector<T *> filtered_list(const std::vector<T *> &input_list, const std::vector<bool> &chosen)
  {
    std::vector<T *> final_list;
    for(int iw=0; iw<input_list.size(); iw++)
      if(chosen[iw]) final_list.push_back(input_list[iw]);
    return final_list;
  }

  const std::vector<ParticleSet *> extract_els_list(const std::vector<Mover *> &mover_list)
  {
    std::vector<ParticleSet *> els_list;
    for(auto it=mover_list.begin(); it!=mover_list.end(); it++)
      els_list.push_back(&(*it)->els);
    return els_list;
  }

  const std::vector<WaveFunction *> extract_wf_list(const std::vector<Mover *> &mover_list)
  {
    std::vector<WaveFunction *> wf_list;
    for(auto it=mover_list.begin(); it!=mover_list.end(); it++)
      wf_list.push_back(&(*it)->wavefunction);
    return wf_list;
  }

}

#endif
