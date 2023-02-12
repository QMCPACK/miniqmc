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


#include <Particle/ParticleSet_builder.hpp>
#include <Input/Input.hpp>
#include <vector>

namespace qmcplusplus
{

std::unique_ptr<SimulationCell> global_cell;

std::unique_ptr<ParticleSet> build_ions(const Tensor<int, 3>& tmat, bool use_offload)
{
  auto ions = tile_cell(global_cell, tmat, use_offload);
  ions->setName("ion");
  ions->update();
  return ions;
}

std::unique_ptr<ParticleSet> build_els(const ParticleSet& ions, RandomGenerator<QMCTraits::RealType>& rng, bool use_offload)
{
  auto els_ptr = std::make_unique<ParticleSet>(ions.getSimulationCell(), use_offload ? DynamicCoordinateKind::DC_POS_OFFLOAD : DynamicCoordinateKind::DC_POS);
  auto& els(*els_ptr);

  els.setName("e");
  const int nels  = count_electrons(ions, 1);
  const int nels3 = 3 * nels;

  { // create up/down electrons
    std::vector<int> ud(2);
    ud[0] = nels / 2;
    ud[1] = nels - ud[0];
    els.create(ud);
    els.R.InUnit = 1;
    rng.generate_uniform(&els.R[0][0], nels3);
    els.convert2Cart(els.R); // convert to Cartiesian
    els.update();
  }

  return els_ptr;
}

} // namespace qmcplusplus
