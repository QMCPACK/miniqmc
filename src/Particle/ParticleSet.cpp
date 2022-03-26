////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by:
// Ken Esler, kpesler@gmail.com,
//    University of Illinois at Urbana-Champaign
// Luke Shulenburger, lshulen@sandia.gov,
//    Sandia National Laboratories
// Jeremy McMinnis, jmcminis@gmail.com,
//    University of Illinois at Urbana-Champaign
// Jeongnim Kim, jeongnim.kim@gmail.com,
//    University of Illinois at Urbana-Champaign
// Jaron T. Krogel, krogeljt@ornl.gov,
//    Oak Ridge National Laboratory
// Ye Luo, yeluo@anl.gov,
//    Argonne National Laboratory
// Mark A. Berrill, berrillma@ornl.gov,
//    Oak Ridge National Laboratory
// Mark Dewing, markdewing@gmail.com,
//    University of Illinois at Urbana-Champaign
//
// File created by:
// Jeongnim Kim, jeongnim.kim@gmail.com,
//    University of Illinois at Urbana-Champaign
////////////////////////////////////////////////////////////////////////////////

#include <numeric>
#include <iomanip>
#include "Particle/ParticleSet.h"
#include "Particle/DistanceTableData.h"
#include "Particle/DistanceTable.h"
#include "Utilities/RandomGenerator.h"

/** @file ParticleSet.cpp
 * @brief Particle positions and related data
 */

//#define PACK_DISTANCETABLES

namespace qmcplusplus
{

TimerNameList_t<DistanceTimers> DistanceTimerNames{
    {Timer_makeMove, "Make move"},
    {Timer_setActive, "Set active"},
    {Timer_acceptMove, "Accept move"},
};

ParticleSet::ParticleSet() : myName("none"), IsGrouped(true), SameMass(true), activePtcl(-1)
{
  setup_timers(timers, DistanceTimerNames, timer_level_coarse);
}

ParticleSet::ParticleSet(const ParticleSet& p)
    : IsGrouped(p.IsGrouped), SameMass(true), activePtcl(-1), mySpecies(p.getSpeciesSet())
{
  //distance_timer = timer_manager.createTimer("Distance Tables", timer_level_coarse);
  setup_timers(timers, DistanceTimerNames, timer_level_coarse);
  // initBase();
  assign(p); // only the base is copied, assumes that other properties are not
             // assignable
  // need explicit copy:
  Mass = p.Mass;
  Z    = p.Z;
  this->setName(p.getName());
  app_debug() << "  Copying a particle set " << p.getName() << " to " << this->getName() << " groups=" << groups()
            << std::endl;
  // construct the distance tables with the same order
  if (p.DistTables.size())
  {
    app_debug() << "  Cloning distance tables. It has " << p.DistTables.size() << std::endl;
    addTable(*this); // first is always for this-this pair
    for (int i = 1; i < p.DistTables.size(); ++i)
      addTable(p.DistTables[i]->origin());
  }

  for (int i = 0; i < p.DistTables.size(); ++i)
    DistTables[i]->setFullTableNeeds(p.DistTables[i]->getFullTableNeeds());

  RSoA.resize(TotalNum);
}

ParticleSet::~ParticleSet() { clearDistanceTables(); }

void ParticleSet::create(int numPtcl)
{
  resize(numPtcl);
  GroupID = 0;
  R       = RealType(0);
}

void ParticleSet::create(const std::vector<int>& agroup)
{
  SubPtcl.resize(agroup.size() + 1);
  SubPtcl[0] = 0;
  for (int is = 0; is < agroup.size(); is++)
    SubPtcl[is + 1] = SubPtcl[is] + agroup[is];
  size_t nsum = SubPtcl[agroup.size()];
  resize(nsum);
  TotalNum = nsum;
  int loc  = 0;
  for (int i = 0; i < agroup.size(); i++)
  {
    for (int j = 0; j < agroup[i]; j++, loc++)
      GroupID[loc] = i;
  }
}

void ParticleSet::resetGroups()
{
  int nspecies = mySpecies.getTotalNum();
  if (nspecies == 0)
  {
    APP_ABORT("ParticleSet::resetGroups() Failed. No species exisits");
  }
  int natt = mySpecies.numAttributes();
  int qind = mySpecies.addAttribute("charge");
  if (natt == qind)
  {
    app_debug() << " Missing charge attribute of the SpeciesSet " << myName << " particleset" << std::endl;
    app_debug() << " Assume neutral particles Z=0.0 " << std::endl;
    for (int ig = 0; ig < nspecies; ig++)
      mySpecies(qind, ig) = 0.0;
  }
  for (int iat = 0; iat < Z.size(); iat++)
    Z[iat] = mySpecies(qind, GroupID[iat]);
  natt        = mySpecies.numAttributes();
  int massind = mySpecies.addAttribute("mass");
  if (massind == natt)
  {
    for (int ig = 0; ig < nspecies; ig++)
      mySpecies(massind, ig) = 1.0;
  }
  SameMass  = true;
  double m0 = mySpecies(massind, 0);
  for (int ig = 1; ig < nspecies; ig++)
    SameMass &= (mySpecies(massind, ig) == m0);
  if (SameMass)
    app_debug() << "  All the species have the same mass " << m0 << std::endl;
  else
    app_debug() << "  Distinctive masses for each species " << std::endl;
  for (int iat = 0; iat < Mass.size(); iat++)
    Mass[iat] = mySpecies(massind, GroupID[iat]);
  std::vector<int> ng(nspecies, 0);
  for (int iat = 0; iat < GroupID.size(); iat++)
  {
    if (GroupID[iat] < nspecies)
      ng[GroupID[iat]]++;
    else
      APP_ABORT("ParticleSet::resetGroups() Failed. GroupID is out of bound.");
  }
  SubPtcl.resize(nspecies + 1);
  SubPtcl[0] = 0;
  for (int i = 0; i < nspecies; ++i)
    SubPtcl[i + 1] = SubPtcl[i] + ng[i];
  int membersize = mySpecies.addAttribute("membersize");
  for (int ig = 0; ig < nspecies; ++ig)
    mySpecies(membersize, ig) = ng[ig];
  // orgID=ID;
  // orgGroupID=GroupID;
  int new_id = 0;
  for (int i = 0; i < nspecies; ++i)
    for (int iat = 0; iat < GroupID.size(); ++iat)
      if (GroupID[iat] == i)
        IndirectID[new_id++] = ID[iat];
  IsGrouped = true;
  for (int iat = 0; iat < ID.size(); ++iat)
    IsGrouped &= (IndirectID[iat] == ID[iat]);
  if (IsGrouped)
    app_debug() << "Particles are grouped. Safe to use groups " << std::endl;
  else
    app_debug() << "ID is not grouped. Need to use IndirectID for "
                 "species-dependent operations "
              << std::endl;
}

/// write to a std::ostream
bool ParticleSet::get(std::ostream& os) const
{
  os << "  ParticleSet " << getName() << " : ";
  for (int i = 0; i < SubPtcl.size(); i++)
    os << SubPtcl[i] << " ";
  os << "\n\n    " << TotalNum << "\n\n";
  const int maxParticlesToPrint = 10;
  int numToPrint                = std::min(TotalNum, maxParticlesToPrint);

  for (int i = 0; i < numToPrint; i++)
  {
    os << "    " << mySpecies.speciesName[GroupID[i]] << R[i] << std::endl;
  }

  if (numToPrint < TotalNum)
  {
    os << "    (... and " << (TotalNum - numToPrint) << " more particle positions ...)" << std::endl;
  }

  return true;
}

/// read from std::istream
bool ParticleSet::put(std::istream& is) { return true; }

/// reset member data
void ParticleSet::reset() { app_debug() << "<<<< going to set properties >>>> " << std::endl; }

int ParticleSet::addTable(const ParticleSet& psrc)
{
  if (myName == "none")
    APP_ABORT("ParticleSet::addTable needs a proper name for this particle set.");
  if (DistTables.empty())
  {
    DistTables.reserve(4);
    DistTables.push_back(createDistanceTable(*this));
    // add  this-this pair
    myDistTableMap.clear();
    myDistTableMap[myName] = 0;
    app_debug() << "  ... ParticleSet::addTable Create Table #0 " << DistTables[0]->getName() << std::endl;
    if (psrc.getName() == myName)
      return 0;
  }
  if (psrc.getName() == myName)
  {
    app_debug() << "  ... ParticleSet::addTable Reuse Table #" << 0 << " " << DistTables[0]->getName() << std::endl;
    return 0;
  }
  int tid;
  std::map<std::string, int>::iterator tit(myDistTableMap.find(psrc.getName()));
  if (tit == myDistTableMap.end())
  {
    tid = DistTables.size();
    DistTables.push_back(createDistanceTable(psrc, *this));
    myDistTableMap[psrc.getName()] = tid;
    app_debug() << "  ... ParticleSet::addTable Create Table #" << tid << " " << DistTables[tid]->getName() << std::endl;
  }
  else
  {
    tid = (*tit).second;
    app_debug() << "  ... ParticleSet::addTable Reuse Table #" << tid << " " << DistTables[tid]->getName() << std::endl;
  }
  app_debug().flush();
  return tid;
}

void ParticleSet::update(bool skipSK)
{
  RSoA.copyIn(R);
  for (int i = 0; i < DistTables.size(); i++)
    DistTables[i]->evaluate(*this);
  activePtcl = -1;
}

void ParticleSet::setActive(int iat)
{
  ScopedTimer local_timer(timers[Timer_setActive]);

  for (size_t i = 0; i < DistTables.size(); i++)
    DistTables[i]->evaluate(*this, iat);
}

void ParticleSet::flex_setActive(const std::vector<ParticleSet*>& P_list, int iat) const
{
  if (P_list.size() > 1)
  {
    ScopedTimer local_timer(timers[Timer_setActive]);
    for (size_t i = 0; i < DistTables.size(); i++)
    {
#pragma omp parallel for
      for (int iw = 0; iw < P_list.size(); iw++)
        P_list[iw]->DistTables[i]->evaluate(*P_list[iw], iat);
    }
  }
  else if (P_list.size() == 1)
    P_list[0]->setActive(iat);
}

/** move the iat-th particle by displ
 *
 * @param iat the particle that is moved on a sphere
 * @param displ displacement from the current position
 */
void ParticleSet::makeMove(Index_t iat, const SingleParticlePos_t& displ)
{
  ScopedTimer local_timer(timers[Timer_makeMove]);

  activePtcl = iat;
  activePos  = R[iat] + displ;
  for (int i = 0; i < DistTables.size(); ++i)
    DistTables[i]->move(*this, activePos, iat);
}

void ParticleSet::flex_makeMove(const std::vector<ParticleSet*>& P_list,
                                Index_t iat,
                                const std::vector<SingleParticlePos_t>& displs) const
{
  if (P_list.size() > 1)
  {
    ScopedTimer local_timer(timers[Timer_makeMove]);

    std::vector<SingleParticlePos_t> new_positions;
    new_positions.reserve(displs.size());

    for (int iw = 0; iw < P_list.size(); iw++)
    {
      P_list[iw]->activePtcl = iat;
      P_list[iw]->activePos  = P_list[iw]->R[iat] + displs[iw];
      new_positions.push_back(P_list[iw]->activePos);
    }

    auto& p_leader = *P_list[0];
    const int dist_tables_size = p_leader.DistTables.size();
    for (int i = 0; i < dist_tables_size; ++i)
    {
      const auto dt_list(extractDTRefList(P_list, i));
      p_leader.DistTables[i]->mw_move(dt_list, P_list, new_positions, iat);
    }
  }
  else if (P_list.size() == 1)
    P_list[0]->makeMove(iat, displs[0]);
}

/** update the particle attribute by the proposed move
 *@param iat the particle index
 *
 *When the activePtcl is equal to iat, overwrite the position and update the
 *content of the distance tables.
 */
void ParticleSet::acceptMove(Index_t iat)
{
  ScopedTimer local_timer(timers[Timer_acceptMove]);

  if (iat == activePtcl)
  {
    // Update position + distance-table
    for (int i = 0, n = DistTables.size(); i < n; i++)
      DistTables[i]->update(iat);

    R[iat]     = activePos;
    RSoA(iat)  = activePos;
    activePtcl = -1;
  }
  else
  {
    std::ostringstream o;
    o << "  Illegal acceptMove " << iat << " != " << activePtcl;
    APP_ABORT(o.str());
  }
}

void ParticleSet::rejectMove(Index_t iat) { activePtcl = -1; }

void ParticleSet::donePbyP(bool skipSK) { activePtcl = -1; }

void ParticleSet::loadWalker(Walker_t& awalker, bool pbyp)
{
  R = awalker.R;
  RSoA.copyIn(R);
  if (pbyp)
  {
    // in certain cases, full tables must be ready
    for (int i = 0; i < DistTables.size(); i++)
      if (DistTables[i]->getFullTableNeeds())
        DistTables[i]->evaluate(*this);
  }
}

void ParticleSet::saveWalker(Walker_t& awalker) { awalker.R = R; }

void ParticleSet::clearDistanceTables()
{
  // Physically remove the tables
  for (auto iter = DistTables.begin(); iter != DistTables.end(); iter++)
    delete *iter;
  DistTables.clear();
}

const std::vector<ParticleSet::ParticleGradient_t*> extract_G_list(const std::vector<ParticleSet*>& P_list)
{
  std::vector<ParticleSet::ParticleGradient_t*> G_list;
  for (auto it = P_list.begin(); it != P_list.end(); it++)
    G_list.push_back(&(*it)->G);
  return G_list;
}

const std::vector<ParticleSet::ParticleLaplacian_t*> extract_L_list(const std::vector<ParticleSet*>& P_list)
{
  std::vector<ParticleSet::ParticleLaplacian_t*> L_list;
  for (auto it = P_list.begin(); it != P_list.end(); it++)
    L_list.push_back(&(*it)->L);
  return L_list;
}

std::vector<DistanceTableData*> ParticleSet::extractDTRefList(const std::vector<ParticleSet*>& p_list, int id)
{
  std::vector<DistanceTableData*> dt_list;
  dt_list.reserve(p_list.size());
  for (ParticleSet* p : p_list)
    dt_list.push_back(p->DistTables[id]);
  return dt_list;
}

} // namespace qmcplusplus
