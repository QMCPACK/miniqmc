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

//#define PACK_DISTANCETABLES

namespace qmcplusplus
{

ParticleSet::ParticleSet()
    : UseBoundBox(true), IsGrouped(true), myName("none"), SameMass(true),
      myTwist(0.0)
{
}

ParticleSet::ParticleSet(const ParticleSet &p)
    : UseBoundBox(p.UseBoundBox), IsGrouped(p.IsGrouped),
      mySpecies(p.getSpeciesSet()), SameMass(true), myTwist(0.0)
{
  // initBase();
  assign(p); // only the base is copied, assumes that other properties are not
             // assignable
  // need explicit copy:
  Mass = p.Mass;
  Z    = p.Z;
  this->setName(p.getName());
  app_log() << "  Copying a particle set " << p.getName() << " to "
            << this->getName() << " groups=" << groups() << std::endl;
  // construct the distance tables with the same order
  if (p.DistTables.size())
  {
    app_log() << "  Cloning distance tables. It has " << p.DistTables.size()
              << std::endl;
    addTable(*this,
             p.DistTables[0]->DTType); // first is always for this-this pair
    for (int i = 1; i < p.DistTables.size(); ++i)
      addTable(p.DistTables[i]->origin(), p.DistTables[i]->DTType);
  }
  for (int i = 0; i < p.DistTables.size(); ++i)
  {
    DistTables[i]->Need_full_table_loadWalker =
        p.DistTables[i]->Need_full_table_loadWalker;
    DistTables[i]->Rmax = p.DistTables[i]->Rmax;
  }
  myTwist = p.myTwist;

  RSoA.resize(TotalNum);
}

ParticleSet::~ParticleSet() { clearDistanceTables(); }

void ParticleSet::create(int numPtcl)
{
  resize(numPtcl);
  GroupID = 0;
  R       = RealType(0);
}

void ParticleSet::create(const std::vector<int> &agroup)
{
  SubPtcl.resize(agroup.size() + 1);
  SubPtcl[0] = 0;
  for (int is       = 0; is < agroup.size(); is++)
    SubPtcl[is + 1] = SubPtcl[is] + agroup[is];
  size_t nsum       = SubPtcl[agroup.size()];
  resize(nsum);
  TotalNum = nsum;
  int loc  = 0;
  for (int i = 0; i < agroup.size(); i++)
  {
    for (int j     = 0; j < agroup[i]; j++, loc++)
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
    app_log() << " Missing charge attribute of the SpeciesSet " << myName
              << " particleset" << std::endl;
    app_log() << " Assume neutral particles Z=0.0 " << std::endl;
    for (int ig = 0; ig < nspecies; ig++)
      mySpecies(qind, ig) = 0.0;
  }
  for (int iat = 0; iat < Z.size(); iat++)
    Z[iat]     = mySpecies(qind, GroupID[iat]);
  natt         = mySpecies.numAttributes();
  int massind  = mySpecies.addAttribute("mass");
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
    app_log() << "  All the species have the same mass " << m0 << std::endl;
  else
    app_log() << "  Distinctive masses for each species " << std::endl;
  for (int iat = 0; iat < Mass.size(); iat++)
    Mass[iat]  = mySpecies(massind, GroupID[iat]);
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
  for (int i       = 0; i < nspecies; ++i)
    SubPtcl[i + 1] = SubPtcl[i] + ng[i];
  int membersize   = mySpecies.addAttribute("membersize");
  for (int ig = 0; ig < nspecies; ++ig)
    mySpecies(membersize, ig) = ng[ig];
  // orgID=ID;
  // orgGroupID=GroupID;
  int new_id = 0;
  for (int i = 0; i < nspecies; ++i)
    for (int iat = 0; iat < GroupID.size(); ++iat)
      if (GroupID[iat] == i) IndirectID[new_id++] = ID[iat];
  IsGrouped                                       = true;
  for (int iat = 0; iat < ID.size(); ++iat)
    IsGrouped &= (IndirectID[iat] == ID[iat]);
  if (IsGrouped)
    app_log() << "Particles are grouped. Safe to use groups " << std::endl;
  else
    app_log() << "ID is not grouped. Need to use IndirectID for "
                 "species-dependent operations "
              << std::endl;
}

/// write to a std::ostream
bool ParticleSet::get(std::ostream &os) const
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
    os << "    (... and " << (TotalNum - numToPrint)
       << " more particle positions ...)" << std::endl;
  }

  return true;
}

/// read from std::istream
bool ParticleSet::put(std::istream &is) { return true; }

/// reset member data
void ParticleSet::reset()
{
  app_log() << "<<<< going to set properties >>>> " << std::endl;
}

void ParticleSet::setBoundBox(bool yes) { UseBoundBox = yes; }

int ParticleSet::addTable(const ParticleSet &psrc, int dt_type)
{
  if (myName == "none")
    APP_ABORT(
        "ParticleSet::addTable needs a proper name for this particle set.");
  if (DistTables.empty())
  {
    DistTables.reserve(4);
    DistTables.push_back(createDistanceTable(*this, dt_type));
    // add  this-this pair
    myDistTableMap.clear();
    myDistTableMap[myName] = 0;
    app_log() << "  ... ParticleSet::addTable Create Table #0 "
              << DistTables[0]->Name << std::endl;
    DistTables[0]->ID = 0;
    if (psrc.getName() == myName) return 0;
  }
  if (psrc.getName() == myName)
  {
    app_log() << "  ... ParticleSet::addTable Reuse Table #" << 0 << " "
              << DistTables[0]->Name << std::endl;
    // if(!DistTables[0]->is_same_type(dt_type))
    //{//itself is special, cannot mix them: some of the users do not check the
    // index
    //  APP_ABORT("ParticleSet::addTable for itself Cannot mix AoS and SoA
    //  distance tables.\n");
    //}
    return 0;
  }
  int tid;
  std::map<std::string, int>::iterator tit(myDistTableMap.find(psrc.getName()));
  if (tit == myDistTableMap.end())
  {
    tid = DistTables.size();
    DistTables.push_back(createDistanceTable(psrc, *this, dt_type));
    myDistTableMap[psrc.getName()] = tid;
    DistTables[tid]->ID            = tid;
    app_log() << "  ... ParticleSet::addTable Create Table #" << tid << " "
              << DistTables[tid]->Name << std::endl;
  }
  else
  {
    tid = (*tit).second;
    if (dt_type == DT_SOA_PREFERRED ||
        DistTables[tid]->is_same_type(dt_type)) // good to reuse
    {
      app_log() << "  ... ParticleSet::addTable Reuse Table #" << tid << " "
                << DistTables[tid]->Name << std::endl;
    }
    else
    {
      APP_ABORT(
          "ParticleSet::addTable Cannot mix AoS and SoA distance tables.\n");
    }
    // if(dt_type == DT_SOA || dt_type == DT_AOS) //not compatible
    //{
    //}
    // for DT_SOA_PREFERRED or DT_AOS_PREFERRED, return the existing table
  }
  app_log().flush();
  return tid;
}

void ParticleSet::update(bool skipSK)
{
  RSoA.copyIn(R);
  for (int i = 0; i < DistTables.size(); i++)
    DistTables[i]->evaluate(*this);
  Ready4Measure = true;
}

void ParticleSet::setActive(int iat)
{
  for (size_t i = 0, n = DistTables.size(); i < n; i++)
    DistTables[i]->evaluate(*this, iat);
}

/** move a particle iat
 * @param iat the index of the particle to be moved
 * @param displ the displacement of the iath-particle position
 * @return the proposed position
 *
 * Update activePtcl index and activePos position for the proposed move.
 * Evaluate the related distance table data DistanceTableData::Temp.
 */
bool ParticleSet::makeMoveAndCheck(Index_t iat,
                                   const SingleParticlePos_t &displ)
{
  activePtcl = iat;
  // SingleParticlePos_t red_displ(Lattice.toUnit(displ));
  if (UseBoundBox)
  {
    if (Lattice.outOfBound(Lattice.toUnit(displ)))
    {
      return false;
    }
    activePos = R[iat]; // save the current position
    SingleParticlePos_t newpos(activePos + displ);
    newRedPos = Lattice.toUnit(newpos);
    if (Lattice.isValid(newRedPos))
    {
      for (int i = 0; i < DistTables.size(); ++i)
        DistTables[i]->move(*this, newpos, iat);
      R[iat] = newpos;
      return true;
    }
    // out of bound
    return false;
  }
  else
  {
    activePos = R[iat]; // save the current position
    SingleParticlePos_t newpos(activePos + displ);
    for (int i = 0; i < DistTables.size(); ++i)
      DistTables[i]->move(*this, newpos, iat);
    R[iat] = newpos;
    return true;
  }
}

/** move the iat-th particle by displ
 *
 * @param iat the particle that is moved on a sphere
 * @param displ displacement from the current position
 */
void ParticleSet::makeMoveOnSphere(Index_t iat,
                                   const SingleParticlePos_t &displ)
{
  activePtcl = iat;
  activePos  = R[iat]; // save the current position
  SingleParticlePos_t newpos(activePos + displ);
  for (int i = 0; i < DistTables.size(); ++i)
    DistTables[i]->moveOnSphere(*this, newpos, iat);
  R[iat] = newpos;
}

/** update the particle attribute by the proposed move
 *@param iat the particle index
 *
 *When the activePtcl is equal to iat, overwrite the position and update the
 *content of the distance tables.
 */
void ParticleSet::acceptMove(Index_t iat)
{
  if (iat == activePtcl)
  {
    // Update position + distance-table
    for (int i = 0, n = DistTables.size(); i < n; i++)
      DistTables[i]->update(iat);

    RSoA(iat) = R[iat];
  }
  else
  {
    std::ostringstream o;
    o << "  Illegal acceptMove " << iat << " != " << activePtcl;
    APP_ABORT(o.str());
  }
}

void ParticleSet::rejectMove(Index_t iat)
{
  // restore the position by the saved activePos
  R[iat] = activePos;
  for (int i                  = 0; i < DistTables.size(); ++i)
    DistTables[i]->activePtcl = -1;
}

void ParticleSet::donePbyP(bool skipSK)
{
  for (size_t i = 0, nt = DistTables.size(); i < nt; i++)
    DistTables[i]->donePbyP();
  Ready4Measure = true;
}

void ParticleSet::loadWalker(Walker_t &awalker, bool pbyp)
{
  R = awalker.R;
#if defined(ENABLE_AA_SOA)
  RSoA.copyIn(R);
#endif
  if (pbyp)
  {
    // in certain cases, full tables must be ready
    for (int i = 0; i < DistTables.size(); i++)
      if (DistTables[i]->Need_full_table_loadWalker)
        DistTables[i]->evaluate(*this);
  }

  Ready4Measure = false;
}

void ParticleSet::saveWalker(Walker_t &awalker)
{
  awalker.R = R;
  // awalker.DataSet.rewind();
}

void ParticleSet::clearDistanceTables()
{
  // Physically remove the tables
  for (auto iter = DistTables.begin(); iter != DistTables.end(); iter++)
    delete *iter;
  DistTables.clear();
}
}
