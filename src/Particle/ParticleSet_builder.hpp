////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by:
// Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//
// File created by:
// Ye Luo, yeluo@anl.gov, Argonne National Laboratory
////////////////////////////////////////////////////////////////////////////////

#ifndef QMCPLUSPLUS_PARTICLESET_BUILDER_HPP
#define QMCPLUSPLUS_PARTICLESET_BUILDER_HPP

#include <Particle/ParticleSet.h>
#include <Utilities/RandomGenerator.h>

namespace qmcplusplus
{
  /// build the ParticleSet of ions
  int build_ions(ParticleSet &ions,
                 const Tensor<int, 3> &tmat,
                 Tensor<QMCTraits::RealType, 3> &lattice);

  /// build the ParticleSet of electrons
  int build_els(ParticleSet &els,
                const ParticleSet &ions,
                RandomGenerator<QMCTraits::RealType> &rng);
}

#endif
