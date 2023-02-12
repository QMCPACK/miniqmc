//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2021 QMCPACK developers.
//
// File developed by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//
// File created by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//////////////////////////////////////////////////////////////////////////////////////


#ifndef QMCPLUSPLUS_SIMULATIONCELL_H
#define QMCPLUSPLUS_SIMULATIONCELL_H

#include "Utilities/Configuration.h"

namespace qmcplusplus
{

class SimulationCell
{
public:
  using Lattice = PtclOnLatticeTraits::ParticleLayout;

  SimulationCell() = default;
  SimulationCell(const Lattice& lattice)
    : lattice_(lattice), primative_lattice_(lattice) {}

  const Lattice& getLattice() const { return lattice_; }
  const Lattice& getPrimLattice() const { return primative_lattice_; }

private:
  ///simulation cell lattice
  Lattice lattice_;
  ///Primative cell lattice
  Lattice primative_lattice_;
};
} // namespace qmcplusplus
#endif
