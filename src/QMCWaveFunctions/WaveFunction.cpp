////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by:
//
// File created by: Jeongnim Kim, jeongnim.kim@intel.com, Intel Corp.
////////////////////////////////////////////////////////////////////////////////
// -*- C++ -*-

/*!
 * @file WaveFunction.cpp
   @brief Wavefunction based on Structure of Arrays (SoA) storage
 */

#include <QMCWaveFunctions/WaveFunction.h>
#include <QMCWaveFunctions/Determinant.h>
#include <QMCWaveFunctions/Jastrow/BsplineFunctor.h>
#include <QMCWaveFunctions/Jastrow/PolynomialFunctor3D.h>
#include <QMCWaveFunctions/Jastrow/OneBodyJastrow.h>
#include <QMCWaveFunctions/Jastrow/TwoBodyJastrow.h>
#include <QMCWaveFunctions/Jastrow/ThreeBodyJastrow.h>
#include <Input/Input.hpp>

namespace qmcplusplus
{

  /*
enum WaveFunctionTimers
{
  Timer_Det,
  Timer_GL,
};
  */

TimerNameLevelList_t<WaveFunctionTimers> WaveFunctionTimerNames =
    {{Timer_Det, "Determinant", timer_level_fine}, {Timer_GL, "Kinetic Energy", timer_level_coarse}};


void build_WaveFunction(bool useRef,
                        WaveFunction& WF,
                        ParticleSet& ions,
                        ParticleSet& els,
                        const RandomGenerator<QMCTraits::RealType>& RNG,
                        bool enableJ3)
{
  using valT = WaveFunction::valT;
  using posT = WaveFunction::posT;

  if (WF.Is_built)
  {
    app_log() << "The wavefunction was built before!" << std::endl;
    return;
  }

  const int nelup = els.getTotalNum() / 2;

  Kokkos::Profiling::pushRegion("building Wavefunction");
  using J1OrbType = OneBodyJastrow<BsplineFunctor<valT>>;
  using J2OrbType = TwoBodyJastrow<BsplineFunctor<valT>>;
  using J3OrbType = ThreeBodyJastrow<PolynomialFunctor3D>;
  using DetType   = DiracDeterminant;
  
  ions.RSoA = ions.R;
  els.RSoA  = els.R;
  
  // distance tables
  els.addTable(els, DT_SOA);
  WF.ei_TableID = els.addTable(ions, DT_SOA);
  
  // determinant component
  WF.nelup  = nelup;
  Kokkos::Profiling::pushRegion("creating determinants");
  WF.Det_up = new DetType(nelup, RNG, 0);
  WF.Det_dn = new DetType(els.getTotalNum() - nelup, RNG, nelup);
  Kokkos::Profiling::popRegion();
  
  // J1 component
  Kokkos::Profiling::pushRegion("creating J1");
  J1OrbType* J1 = new J1OrbType(ions, els);
  buildJ1(*J1, els.Lattice.WignerSeitzRadius);
  WF.Jastrows.push_back(J1);
  Kokkos::Profiling::popRegion();

  // J2 component
  Kokkos::Profiling::pushRegion("creating J2");
  J2OrbType* J2 = new J2OrbType(els);
  buildJ2(*J2, els.Lattice.WignerSeitzRadius);
  //std::cout << "finished buildJ2" << std::endl;
  WF.Jastrows.push_back(J2);
  Kokkos::Profiling::popRegion();
  
  // J3 component
  if (enableJ3)
  {
    J3OrbType* J3 = new J3OrbType(ions, els);
    buildJeeI(*J3, els.Lattice.WignerSeitzRadius);
    WF.Jastrows.push_back(J3);
  }
  Kokkos::Profiling::popRegion();

  WF.setupTimers();
  WF.Is_built = true;
}

WaveFunction::~WaveFunction()
{
  if (Is_built)
  {
    delete Det_up;
    delete Det_dn;
    for (size_t i = 0; i < Jastrows.size(); i++)
      delete Jastrows[i];
  }
}

void WaveFunction::setupTimers()
{
  setup_timers(timers, WaveFunctionTimerNames);
  for (int i = 0; i < Jastrows.size(); i++)
  {
    jastrow_timers.push_back(
        TimerManager.createTimer(Jastrows[i]->WaveFunctionComponentName, timer_level_fine));
  }
}


void WaveFunction::multi_evaluateLog(const std::vector<WaveFunction*>& WF_list,
				     WaveFunctionKokkos& wfc,
				     Kokkos::View<ParticleSet::pskType*>& psk) const
{
  Kokkos::Profiling::pushRegion("multi_evaluateLog (initialization)");
  const int numItems = WF_list.size();
  if (WF_list[0]->FirstTime)
  {
    constexpr valT czero(0);
    ParticleSet::ParticleValue_t LogValues(numItems);

    if (numItems > 0) {
      // Determinants
      std::vector<WaveFunctionComponent*> up_list(extract_up_list(WF_list));
      doDiracDeterminantMultiEvaluateLog(wfc.upDets, up_list, LogValues);
      for (int iw = 0; iw < numItems; iw++)
	WF_list[iw]->LogValue = LogValues[iw];
      
      std::vector<WaveFunctionComponent*> dn_list(extract_dn_list(WF_list));
      doDiracDeterminantMultiEvaluateLog(wfc.downDets, dn_list, LogValues);
      for (int iw = 0; iw < numItems; iw++)
	WF_list[iw]->LogValue += LogValues[iw];

      // Jastrow factors      
      for (size_t i = 0; i < Jastrows.size(); i++)
	{
	  std::vector<WaveFunctionComponent*> jas_list(extract_jas_list(WF_list, i));
	  Jastrows[i]->multi_evaluateLog(jas_list, wfc, psk, LogValues);
	  for (int iw = 0; iw < numItems; iw++)
	    WF_list[iw]->LogValue += LogValues[iw];
	}
      for (int iw = 0; iw < numItems; iw++)
	WF_list[iw]->FirstTime = false;
    }
  }
  Kokkos::Profiling::popRegion();
}

void WaveFunction::multi_evalGrad(const std::vector<WaveFunction*>& WF_list,
				  WaveFunctionKokkos& wfc,
				  Kokkos::View<ParticleSet::pskType*>& psk,
				  int iat,
				  std::vector<posT>& grad_now) const
{

  const int numItems = WF_list.size();
  if (numItems > 0) {
    timers[Timer_Det]->start();
    std::vector<posT> grad_now_det(numItems);
    if (iat < nelup)
      {
	std::vector<WaveFunctionComponent*> up_list(extract_up_list(WF_list));
	Det_up->multi_evalGrad(up_list, wfc, psk, iat, grad_now_det);
      }
    else
      {
	std::vector<WaveFunctionComponent*> dn_list(extract_dn_list(WF_list));
	Det_dn->multi_evalGrad(dn_list, wfc, psk, iat, grad_now_det);
      }
    for (int iw = 0; iw < numItems; iw++)
      grad_now[iw] = grad_now_det[iw];
    timers[Timer_Det]->stop();
    
    for (size_t i = 0; i < Jastrows.size(); i++)
      {
	jastrow_timers[i]->start();
	std::vector<posT> grad_now_jas(numItems);
	std::vector<WaveFunctionComponent*> jas_list(extract_jas_list(WF_list, i));
	Jastrows[i]->multi_evalGrad(jas_list, wfc, psk, iat, grad_now_jas);
	
	for (int iw = 0; iw < numItems; iw++)
	  grad_now[iw] += grad_now_jas[iw];
	jastrow_timers[i]->stop();
      }
  }
}


void WaveFunction::multi_acceptrestoreMove(const std::vector<WaveFunction*>& WF_list,
					   WaveFunctionKokkos& wfc,
					   Kokkos::View<ParticleSet::pskType*>& psk,
					   Kokkos::View<int*>& isAcceptedMap,
					   int numAccepted, int iel) const {
  if (numAccepted > 0) {
    timers[Timer_Det]->start();
    if (iel < nelup)
      {
	std::vector<WaveFunctionComponent*> up_list(extract_up_list(WF_list));
	doDiracDeterminantMultiAccept(wfc.upDets, up_list, isAcceptedMap, numAccepted, iel);
      }
    else
      {
	std::vector<WaveFunctionComponent*> dn_list(extract_dn_list(WF_list));
	doDiracDeterminantMultiAccept(wfc.downDets, dn_list, isAcceptedMap, numAccepted, iel);
      }
    timers[Timer_Det]->stop();

    for (size_t i = 0; i < Jastrows.size(); i++)
      {
	jastrow_timers[i]->start();
	std::vector<WaveFunctionComponent*> jas_list(extract_jas_list(WF_list, i));
	Jastrows[i]->multi_acceptrestoreMove(jas_list,wfc, psk, isAcceptedMap, numAccepted, iel);
	jastrow_timers[i]->stop();
      }
  }
}


void WaveFunction::multi_evaluateGL(const std::vector<WaveFunction*>& WF_list,
                                    const std::vector<ParticleSet*>& P_list) const
{
  constexpr valT czero(0);
  const std::vector<ParticleSet::ParticleGradient_t*> G_list(extract_G_list(P_list));
  const std::vector<ParticleSet::ParticleLaplacian_t*> L_list(extract_L_list(P_list));

  for (int iw = 0; iw < P_list.size(); iw++)
  {
    *G_list[iw] = czero;
    *L_list[iw] = czero;
  }
  // det up/dn
 
  if (G_list.size() > 0) {
    std::vector<WaveFunctionComponent*> up_list(extract_up_list(WF_list));
    Det_up->multi_evaluateGL(up_list, P_list, G_list, L_list);
    for (int iw = 0; iw < P_list.size(); iw++)
      WF_list[iw]->LogValue = up_list[iw]->LogValue;
    std::vector<WaveFunctionComponent*> dn_list(extract_dn_list(WF_list));
    Det_dn->multi_evaluateGL(dn_list, P_list, G_list, L_list);
    for (int iw = 0; iw < P_list.size(); iw++)
      WF_list[iw]->LogValue += dn_list[iw]->LogValue;
    // Jastrow factors
    for (size_t i = 0; i < Jastrows.size(); i++)
      {
	std::vector<WaveFunctionComponent*> jas_list(extract_jas_list(WF_list, i));
	Jastrows[i]->multi_evaluateGL(jas_list, P_list, G_list, L_list);
	for (int iw = 0; iw < P_list.size(); iw++)
	  WF_list[iw]->LogValue += jas_list[iw]->LogValue;
      }
  }
}

const std::vector<WaveFunctionComponent*> extract_up_list(const std::vector<WaveFunction*>& WF_list)
{
  std::vector<WaveFunctionComponent*> up_list;
  for (auto it = WF_list.begin(); it != WF_list.end(); it++)
    up_list.push_back((*it)->Det_up);
  return up_list;
}

const std::vector<WaveFunctionComponent*> extract_dn_list(const std::vector<WaveFunction*>& WF_list)
{
  std::vector<WaveFunctionComponent*> dn_list;
  for (auto it = WF_list.begin(); it != WF_list.end(); it++)
    dn_list.push_back((*it)->Det_dn);
  return dn_list;
}

const std::vector<WaveFunctionComponent*>
    extract_jas_list(const std::vector<WaveFunction*>& WF_list, int jas_id)
{
  std::vector<WaveFunctionComponent*> jas_list;
  for (auto it = WF_list.begin(); it != WF_list.end(); it++)
    jas_list.push_back((*it)->Jastrows[jas_id]);
  return jas_list;
}

} // namespace qmcplusplus
