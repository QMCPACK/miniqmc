//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by: Ken Esler, kpesler@gmail.com, University of Illinois at Urbana-Champaign
//                    Raymond Clay III, j.k.rofling@gmail.com, Lawrence Livermore National Laboratory
//                    Jeremy McMinnis, jmcminis@gmail.com, University of Illinois at Urbana-Champaign
//                    Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//                    Jaron T. Krogel, krogeljt@ornl.gov, Oak Ridge National Laboratory
//                    Mark A. Berrill, berrillma@ornl.gov, Oak Ridge National Laboratory
//
// File created by: Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//////////////////////////////////////////////////////////////////////////////////////
    
    
#include "QMCWaveFunctions/OrbitalBase.h"
//#include "QMCWaveFunctions/ProxyOrbital.h"

namespace qmcplusplus
{
OrbitalBase::OrbitalBase():
  IsOptimizing(false), Optimizable(true), UpdateMode(ORB_WALKER), //UseBuffer(true), //Counter(0),
  LogValue(1.0),PhaseValue(0.0),OrbitalName("OrbitalBase"), derivsDone(false), parameterType(0)
#if !defined(ENABLE_SMARTPOINTER)
  , ionDerivs(false)
#endif
{ 
  ///store instead of computing
  Need2Compute4PbyP=false;
}

// OrbitalBase::OrbitalBase(const OrbitalBase& old):
//   Optimizable(old.Optimizable), UseBuffer(old.UseBuffer),
//   dPsi(old.dPsi),dLogPsi(old.dLogPsi),d2LogPsi(old.d2LogPsi),
//   OrbitalName(old.OrbitalName),myVars(old.myVars)
// {
//   //
//   //if(dLogPsi.size()) dLogPsi.resize(dLogPsi.size());
//   //if(d2LogPsi.size()) dLogPsi.resize(d2LogPsi.size());
//   //if(dPsi) dPsi=old.dPsi->makeClone();
// }

///** makeClone uses optVars  to determine if it will make a clone (deepcopy)
// * or make a ProxyOrbital.
// */
//OrbitalBasePtr OrbitalBase::makeClone(ParticleSet& tpq,  int i)
//{
//  int loc=-1;
//  int iii=0;
//  while(loc<0 && iii<myVars.size())
//  {
//    if(myVars.Index[iii]==i) loc=iii;
//    ++iii;
//  }
//  if(loc<0)
//    return makeProxy(tpq,this);
//  else
//    return makeClone(tpq,true);
//}

/*@todo makeClone should be a pure virtual function
 */
OrbitalBasePtr OrbitalBase::makeClone(ParticleSet& tpq) const
{
  APP_ABORT("Implement OrbitalBase::makeClone "+OrbitalName+ " class.");
  return 0;
}

//void OrbitalBase::copyFrom(const OrbitalBase& old)
//{
//  APP_ABORT("OrbitalBase::copyFrom needs to be implemented by a derived class.");
//}

OrbitalBasePtr OrbitalBase::makeProxy(ParticleSet& tpq)
{
  return 0;
//    OrbitalBase* proxy= new ProxyOrbital(tpq,org);
//#if defined(ENABLE_SMARTPOINTER)
//    return OrbitalBasePtr(proxy);
//#else
//    return proxy;
//#endif
}

OrbitalBase::RealType OrbitalBase::KECorrection()
{
  return 0;
}

/** done PbyP update, prepare for the measurements */
void OrbitalBase::updateAfterSweep(ParticleSet& P,
    ParticleSet::ParticleGradient_t& G,
    ParticleSet::ParticleLaplacian_t& L)
{
  std::ostringstream o;
  o << "OrbitalBase::updateAfterSweep is not implemented by " << OrbitalName;
  APP_ABORT(o.str());
}

void OrbitalBase::get_ratios(ParticleSet& P, std::vector<ValueType>& ratios)
{
  std::ostringstream o;
  o << "OrbitalBase::get_ratios is not implemented by " << OrbitalName;
  APP_ABORT(o.str());
}

void OrbitalBase::evaluateRatios(VirtualParticleSet& P, std::vector<ValueType>& ratios)
{
  std::ostringstream o;
  o << "OrbitalBase::evaluateRatios is not implemented by " << OrbitalName;
  APP_ABORT(o.str());
}

}

