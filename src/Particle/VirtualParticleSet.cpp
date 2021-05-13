//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//                    Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//
// File created by: Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//////////////////////////////////////////////////////////////////////////////////////


/** @file VirtualParticleSet.cpp
 * A proxy class to the quantum ParticleSet
 */

#include <Particle/VirtualParticleSet.h>
#include <Particle/DistanceTableData.h>
#include <Particle/DistanceTable.h>

namespace qmcplusplus
{
VirtualParticleSet::VirtualParticleSet(const ParticleSet& p, int nptcl) : refPS(p)
{
  setName("virtual");

  //initialize local data structure
  Lattice  = p.Lattice;
  TotalNum = nptcl;
  R.resize(nptcl);
  RSoA.resize(nptcl);

  //create distancetables
  if (refPS.DistTables.size())
  {
    DistTables.resize(refPS.DistTables.size());
    for (int i = 0; i < DistTables.size(); ++i)
      DistTables[i] = createDistanceTable(refPS.DistTables[i]->origin(), *this);
  }
}

/// move virtual particles to new postions and update distance tables
void VirtualParticleSet::makeMoves(int jel, const ParticlePos_t& vitualPos, bool sphere, int iat)
{
  ScopedTimer local_timer(timers[Timer_makeMove]);

  if (sphere && iat < 0)
    APP_ABORT("VirtualParticleSet::makeMoves is invoked incorrectly, the flag sphere=true requires iat specified!");
  onSphere = sphere;
  refPtcl       = jel;
  refSourcePtcl = iat;
  R             = vitualPos;
  RSoA.copyIn(R);
  for (int i = 0; i < DistTables.size(); i++)
    DistTables[i]->evaluate(*this);
}

} // namespace qmcplusplus
