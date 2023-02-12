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
extern std::unique_ptr<SimulationCell> global_cell;

/// build the ParticleSet of ions
std::unique_ptr<ParticleSet> build_ions(const Tensor<int, 3>& tmat, bool use_offload);

/// build the ParticleSet of electrons
std::unique_ptr<ParticleSet> build_els(const ParticleSet& ions, RandomGenerator<QMCTraits::RealType>& rng, bool use_offload);
} // namespace qmcplusplus

#endif
