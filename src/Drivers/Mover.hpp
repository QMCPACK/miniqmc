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
  struct Mover
  {
    using RealType = QMCTraits::RealType;

    /// random number generator
    RandomGenerator<RealType> rng;
    /// electrons
    ParticleSet               els;
    /// single particle orbitals
    SPOSet                    *spo;
    /// wavefunction container
    WaveFunction              wavefunction;
    /// non-local pseudo-potentials
    NonLocalPP<RealType>      nlpp;

    /// constructor
    Mover(const uint32_t myPrime,
          const ParticleSet &ions)
      : spo(nullptr), rng(myPrime), nlpp(rng)
    {
      build_els(els, ions, rng);
    }

    /// destructor
    ~Mover() { delete spo; }

  };

}

#endif
