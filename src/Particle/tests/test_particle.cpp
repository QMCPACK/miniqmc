//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2019 QMCPACK developers.
//
// File developed by: Mark Dewing, markdewing@gmail.com, University of Illinois at Urbana-Champaign
//
// File created by: Mark Dewing, markdewing@gmail.com, University of Illinois at Urbana-Champaign
//////////////////////////////////////////////////////////////////////////////////////


#include <stdio.h>
#include <string>

#include "catch.hpp"

#include "Utilities/Configuration.h"
#include "Numerics/OhmmsPETE/OhmmsMatrix.h"
#include "Numerics/OhmmsPETE/TinyVector.h"
#include "Particle/Lattice/CrystalLattice.h"
#include "Particle/Lattice/ParticleBConds.h"
#include "Particle/ParticleSet.h"
#include "Particle/DistanceTable.h"
#include "Particle/DistanceTableData.h"

using std::string;

namespace qmcplusplus
{
TEST_CASE("symmetric_distance_table PBC", "[particle]")
{
  ParticleSet source;

  CrystalLattice<OHMMS_PRECISION, 3, OHMMS_ORTHO> grid;
  grid.BoxBConds = true; // periodic
  grid.R = ParticleSet::Tensor_t(6.74632230, 6.74632230, 0.00000000, 0.00000000, 3.37316115, 3.37316115, 3.37316115,
                                 0.00000000, 3.37316115);
  grid.reset();

  source.setName("electrons");
  source.Lattice.set(grid);

  source.create(4);
  source.R[0] = ParticleSet::PosType(0.00000000, 0.00000000, 0.00000000);
  source.R[1] = ParticleSet::PosType(1.68658058, 1.68658058, 1.68658058);
  source.R[2] = ParticleSet::PosType(3.37316115, 3.37316115, 0.00000000);
  source.R[3] = ParticleSet::PosType(5.05974172, 5.05974172, 1.68658058);

  int TableID = source.addTable(source);
  source.update();

  REQUIRE(source.DistTables[TableID]->Distances[1][2] == Approx(2.9212432441));
  REQUIRE(source.DistTables[TableID]->Distances[2][1] == Approx(2.9212432441));
  REQUIRE(source.DistTables[TableID]->Displacements[1][2][0] == Approx(1.68658057));
  REQUIRE(source.DistTables[TableID]->Displacements[1][2][1] == Approx(1.68658057));
  REQUIRE(source.DistTables[TableID]->Displacements[1][2][2] == Approx(-1.68658058));
  REQUIRE(source.DistTables[TableID]->Displacements[2][1][0] == Approx(-1.68658057));
  REQUIRE(source.DistTables[TableID]->Displacements[2][1][1] == Approx(-1.68658057));
  REQUIRE(source.DistTables[TableID]->Displacements[2][1][2] == Approx(1.68658057));
}

} // namespace qmcplusplus
