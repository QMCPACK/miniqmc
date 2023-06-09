//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by:  Mark Dewing, markdewing@gmail.com, University of Illinois at Urbana-Champaign
//
// File created by: Mark Dewing, markdewing@gmail.com, University of Illinois at Urbana-Champaign
//////////////////////////////////////////////////////////////////////////////////////


#include "catch.hpp"

#include <stdio.h>
#include <string>
#include "tinyxml/tinyxml2.h"
#include "Particle/ParticleSet.h"
#include "Particle/DistanceTable.h"
#include <ResourceCollection.h>

using std::string;

namespace qmcplusplus
{
SimulationCell parse_pbc_fcc_lattice()
{
  CrystalLattice<OHMMS_PRECISION, OHMMS_DIM> lattice;
  lattice.BoxBConds = true;
  lattice.R         = {6., 6., 0., 0., 6., 6., 6., 0., 0.};
  lattice.reset();
  return SimulationCell(lattice);
}

void parse_electron_ion_pbc_z(ParticleSet& ions, ParticleSet& electrons)
{
  // read particle set
  electrons.setName("e");
  electrons.create({2, 1});
  electrons.R[0] = {0., 0., 0.};
  electrons.R[1] = {3., 0., 0.};
  electrons.R[2] = {0., 0., 3.};

  SpeciesSet& tspecies         = electrons.getSpeciesSet();
  int upIdx                    = tspecies.addSpecies("u");
  int downIdx                  = tspecies.addSpecies("d");
  int chargeIdx                = tspecies.addAttribute("charge");
  tspecies(chargeIdx, upIdx)   = -1;
  tspecies(chargeIdx, downIdx) = -1;
  electrons.resetGroups();

  ions.setName("ion0");
  ions.create({8});
  ions.R[0] = {0., 0., 0.};
  ions.R[1] = {3., 0., 0.};
  ions.R[2] = {0., 3., 0.};
  ions.R[3] = {3., 3., 0.};
  ions.R[4] = {0., 0., 3.};
  ions.R[5] = {3., 0., 3.};
  ions.R[6] = {0., 3., 3.};
  ions.R[7] = {3., 3., 3.};

  SpeciesSet& ion_species        = ions.getSpeciesSet();
  int Idx                        = ion_species.addSpecies("H");
  int ionchargeIdx               = ion_species.addAttribute("charge");
  ion_species(ionchargeIdx, Idx) = 1;
  ions.resetGroups();

  REQUIRE(electrons.getName() == "e");
  REQUIRE(electrons.isSameMass());
  REQUIRE(ions.getName() == "ion0");
  REQUIRE(ions.isSameMass());
}

void test_distance_fcc_pbc_z_batched_APIs(DynamicCoordinateKind test_kind)
{
  // test that particle distances are properly calculated under periodic boundary condition
  // There are many details in this example, but the main idea is simple: When a particle is moved by a full lattice vector, no distance should change.

  const SimulationCell simulation_cell(parse_pbc_fcc_lattice());
  ParticleSet ions(simulation_cell), electrons(simulation_cell, test_kind);
  parse_electron_ion_pbc_z(ions, electrons);

  // calculate particle distances
  ions.update();
  const int ee_tid = electrons.addTable(electrons, DTModes::NEED_FULL_TABLE_ON_HOST_AFTER_DONEPBYP);
  // get target particle set's distance table data
  const auto& ee_dtable = electrons.getDistTableAA(ee_tid);
  CHECK(ee_dtable.getName() == "e_e");
  electrons.update();

  // shift electron 0 a bit to avoid box edges.
  ParticleSet::SingleParticlePos shift(0.1, 0.2, -0.1);
  electrons.makeMove(0, shift);
  electrons.accept_rejectMove(0, true, false);
  electrons.donePbyP();

  ParticleSet electrons_clone(electrons);
  RefVectorWithLeader<ParticleSet> p_list(electrons);
  p_list.push_back(electrons);
  p_list.push_back(electrons_clone);

  ResourceCollection pset_res("test_pset_res");
  electrons.createResource(pset_res);
  ResourceCollectionTeamLock<ParticleSet> mw_pset_lock(pset_res, p_list);

  std::vector<ParticleSet::SingleParticlePos> disp{{0.2, 0.1, 0.3}, {0.2, 0.1, 0.3}};

  ParticleSet::mw_makeMove(p_list, 0, disp);
  ParticleSet::mw_accept_rejectMove(p_list, 0, {true, true}, true);
  ParticleSet::mw_makeMove(p_list, 1, disp);
  ParticleSet::mw_accept_rejectMove(p_list, 1, {false, false}, true);

  ParticleSet::mw_donePbyP(p_list);
  CHECK(ee_dtable.getDistRow(1)[0] == Approx(2.7239676944));
  CHECK(ee_dtable.getDisplRow(1)[0][0] == Approx(-2.7));
  CHECK(ee_dtable.getDisplRow(1)[0][1] == Approx(0.3));
  CHECK(ee_dtable.getDisplRow(1)[0][2] == Approx(0.2));
} // test_distance_pbc_z_batched_APIs

TEST_CASE("distance_pbc_z batched APIs", "[distance_table][xml]")
{
  test_distance_fcc_pbc_z_batched_APIs(DynamicCoordinateKind::DC_POS);
  test_distance_fcc_pbc_z_batched_APIs(DynamicCoordinateKind::DC_POS_OFFLOAD);
}

} // namespace qmcplusplus
