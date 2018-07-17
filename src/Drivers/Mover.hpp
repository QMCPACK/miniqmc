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


#ifndef QMCPLUSPLUS_MOVER_HPP
#define QMCPLUSPLUS_MOVER_HPP

#include <Utilities/Configuration.h>
#include <Utilities/RandomGenerator.h>
#include <Particle/ParticleSet.h>
#include <QMCWaveFunctions/einspline_spo.hpp>
#include <Numerics/Spline2/MultiBspline.hpp>
#include <QMCWaveFunctions/WaveFunction.h>
#include <Input/pseudo.hpp>

namespace qmcplusplus
{
  struct Mover
  {
    using RealType = QMCTraits::RealType;
    using spo_type = einspline_spo<RealType, MultiBspline<RealType> >;

    RandomGenerator<RealType> rng;
    ParticleSet               els;
    spo_type                  spo;
    WaveFunction              wavefunction;
    NonLocalPP<RealType>      nlpp;

    Mover(const spo_type &spo_main,
          const int team_size,
          const int member_id,
          const uint32_t myPrime,
          const ParticleSet &ions);
  };

  int build_ions(ParticleSet &ions,
                 const Tensor<int, 3> &tmat,
                 Tensor<QMCTraits::RealType, 3> &lattice);

  int build_els(ParticleSet &els,
                const ParticleSet &ions,
                RandomGenerator<QMCTraits::RealType> &rng);
}

#endif
