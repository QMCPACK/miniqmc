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

/**
 * @file WaveFunction.h
 * @brief Top level wavefunction container
 *
 * Represents a product of wavefunction components (classes based on
 * WaveFunctionComponent).
 *
 * Corresponds to QMCWaveFunction/TrialWaveFunction.h in the QMCPACK source.
 */

#ifndef QMCPLUSPLUS_WAVEFUNCTIONS_H
#define QMCPLUSPLUS_WAVEFUNCTIONS_H
#include <Utilities/Configuration.h>
#include <Utilities/RandomGenerator.h>
#include <Utilities/NewTimer.h>
#include <Particle/ParticleSet.h>
#include <QMCWaveFunctions/WaveFunctionComponent.h>
#include <QMCWaveFunctions/Determinant.h>
#include <QMCWaveFunctions/WaveFunctionKokkos.h>

namespace qmcplusplus
{
/** A minimal TrialWavefunction
 */
enum WaveFunctionTimers
{
  Timer_Det,
  Timer_GL,
};

class WaveFunction
{
  using RealType = OHMMS_PRECISION;
  using valT     = OHMMS_PRECISION;
  using posT     = TinyVector<valT, OHMMS_DIM>;

private:
  std::vector<WaveFunctionComponent*> Jastrows;
  WaveFunctionComponent* Det_up;
  WaveFunctionComponent* Det_dn;
  valT LogValue;

  bool FirstTime, Is_built;
  int nelup, ei_TableID;

  TimerList_t timers;
  TimerList_t jastrow_timers;

public:
  WaveFunction()
      : FirstTime(true),
        Is_built(false),
        nelup(0),
        ei_TableID(1),
        Det_up(nullptr),
        Det_dn(nullptr),
        LogValue(0.0)
  {}
  ~WaveFunction();

  /// operates on a single walker

  /// operates on multiple walkers
  void multi_evaluateLog(const std::vector<WaveFunction*>& WF_list,
                         const std::vector<ParticleSet*>& P_list) const;

  
  void multi_evaluateLog(const std::vector<WaveFunction*>& WF_list,
			 WaveFunctionKokkos& wfc,
			 Kokkos::View<ParticleSet::pskType*>& psk) const;

  void multi_evalGrad(const std::vector<WaveFunction*>& WF_list,
                      const std::vector<ParticleSet*>& P_list,
                      int iat,
                      std::vector<posT>& grad_now) const;

  void multi_evalGrad(const std::vector<WaveFunction*>& WF_list,
                      WaveFunctionKokkos& wfc,
		      Kokkos::View<ParticleSet::pskType*>& psk,
                      int iat,
                      std::vector<posT>& grad_now) const;
  void multi_ratioGrad(const std::vector<WaveFunction*>& WF_list,
                       const std::vector<ParticleSet*>& P_list,
                       int iat,
                       std::vector<valT>& ratio_list,
                       std::vector<posT>& grad_new) const;
  template<typename valsType, typename isValidMapMirrorType>
  void multi_ratioGrad(const std::vector<WaveFunction*>& WF_list,
		       WaveFunctionKokkos wfc,
		       Kokkos::View<ParticleSet::pskType*>& psk,
		       valsType& psiVs,
		       Kokkos::View<int*>& isValidMap, 
		       isValidMapMirrorType& isValidMapMirror, int numValid,
		       int iel, std::vector<valT>& ratio_list,
		       std::vector<posT>& grad_new) const;


  /*
  template<typename apsdType, typename psiVType, typename likeTempRType, typename unlikeTempRType, typename eiListType>
  void multi_ratio(int pairNum, const std::vector<WaveFunction*>& WF_list, apsdType& apsd, psiVType& tempPsiV,
		   likeTempRType& likeTempRs, unlikeTempRType& unlikeTempRs, eiListType& eiList, std::vector<valT>& ratios); 
  */
  template<typename apsdType, typename psiVType, typename likeTempRType, typename unlikeTempRType, typename eiListType>
  void multi_ratio(int pairNum, WaveFunctionKokkos& wfc, apsdType& apsd, psiVType& tempPsiV,
		   likeTempRType& likeTempRs, unlikeTempRType& unlikeTempRs, eiListType& eiList, std::vector<valT>& ratios); 

  /*
  void multi_acceptrestoreMove(const std::vector<WaveFunction*>& WF_list,
                               const std::vector<ParticleSet*>& P_list,
                               const std::vector<bool>& isAccepted,
                               int iat) const;
  */

  void multi_acceptrestoreMove(const std::vector<WaveFunction*>& WF_list,
			       WaveFunctionKokkos& wfc,
			       Kokkos::View<ParticleSet::pskType*>& psk,
			       Kokkos::View<int*>& isAcceptedMap,
			       int numAccepted, int iel) const;

  void multi_evaluateGL(const std::vector<WaveFunction*>& WF_list,
                        const std::vector<ParticleSet*>& P_list) const;

  // others
  int get_ei_TableID() const { return ei_TableID; }
  valT getLogValue() const { return LogValue; }
  void setupTimers();

  // friends
  friend void build_WaveFunction(bool useRef,
                                 WaveFunction& WF,
                                 ParticleSet& ions,
                                 ParticleSet& els,
                                 const RandomGenerator<QMCTraits::RealType>& RNG,
                                 bool enableJ3);
  friend const std::vector<WaveFunctionComponent*>
      extract_up_list(const std::vector<WaveFunction*>& WF_list);
  friend const std::vector<WaveFunctionComponent*>
      extract_dn_list(const std::vector<WaveFunction*>& WF_list);
  friend const std::vector<WaveFunctionComponent*>
      extract_jas_list(const std::vector<WaveFunction*>& WF_list, int jas_id);
};

void build_WaveFunction(bool useRef,
                        WaveFunction& WF,
                        ParticleSet& ions,
                        ParticleSet& els,
                        const RandomGenerator<QMCTraits::RealType>& RNG,
                        bool enableJ3);

const std::vector<WaveFunctionComponent*> extract_up_list(const std::vector<WaveFunction*>& WF_list);
const std::vector<WaveFunctionComponent*> extract_dn_list(const std::vector<WaveFunction*>& WF_list);
const std::vector<WaveFunctionComponent*>
    extract_jas_list(const std::vector<WaveFunction*>& WF_list, int jas_id);

/*
template<typename apsdType, typename psiVType, typename likeTempRType, typename unlikeTempRType, typename eiListType>
void WaveFunction::multi_ratio(int pairNum, const std::vector<WaveFunction*>& WF_list, apsdType& apsd, psiVType& tempPsiV,
			       likeTempRType& likeTempRs, unlikeTempRType& unlikeTempRs, eiListType& eiList,
			       std::vector<valT>& ratios) {
  timers[Timer_Det]->start();
  int numWalkers = eiList.extent(0);
  int numKnots = tempPsiV.extent(1);
  std::vector<valT> ratios_det(numWalkers*numKnots);
  for (int iw = 0; iw < numWalkers*numKnots; iw++) {
    ratios[iw] = valT(0);
  }
  
  auto& tmpEiList = eiList;
  // complication here in that we are not always looking at the same electron from walker to walker
  //std::cout << "about to make EiListMirror" << std::endl;
  //std::cout << "dimensionality of eiList = " << tmpEiList.extent(0) << ", " << tmpEiList.extent(1) << ", " << tmpEiList.extent(2) << std::endl;
  typename eiListType::HostMirror EiListMirror = Kokkos::create_mirror_view(tmpEiList);
  //std::cout << "mirror view created, about to do deep_copy" << std::endl;
  Kokkos::deep_copy(EiListMirror, tmpEiList);
  //std::cout << "finished setting up EiListMirror" << std::endl;

  std::vector<int> packedIndex;
  for (int iw = 0; iw < numWalkers; iw++) {
    const int elNum = EiListMirror(iw,pairNum,0);
    if (elNum > 0) {
      packedIndex.push_back(iw);
    }
  }

  //std::cout << "about to make activeWalkerIndex view" << std::endl;
  Kokkos::View<int*> activeWalkerIndex("activeWalkerIndex", packedIndex.size());
  auto activeWalkerIndexMirror = Kokkos::create_mirror_view(activeWalkerIndex);

  Kokkos::View<DiracDeterminantKokkos*> addk("AllDiracDeterminantKokkos", packedIndex.size());
  auto addkMirror = Kokkos::create_mirror_view(addk);

  for (int i = 0; i< packedIndex.size(); i++) {
    const int awi = packedIndex[i];
    activeWalkerIndexMirror(i) = awi;
    if (EiListMirror(awi,pairNum,0) < nelup) {
      addkMirror(i) = static_cast<DiracDeterminant*>(WF_list[awi]->Det_up)->ddk;
    } else {
      addkMirror(i) = static_cast<DiracDeterminant*>(WF_list[awi]->Det_dn)->ddk;
    }
  }
  Kokkos::deep_copy(activeWalkerIndex, activeWalkerIndexMirror);
  Kokkos::deep_copy(addk, addkMirror);
  // will do this for one set of pairs for every walker if available
  // note, ratios_set holds all the values, so need to index accordingly
  //std::cout << "in multi_ratio, about to call do DiracDeterminantMultiEvalRatio" << std::endl;
  doDiracDeterminantMultiEvalRatio(pairNum, addk, activeWalkerIndex, eiList, tempPsiV, ratios_det);

  for (int i = 0; i < numWalkers*numKnots; i++) {
    ratios[i] = ratios_det[i];
  }
  timers[Timer_Det]->stop();

  for (size_t i = 0; i < Jastrows.size(); i++)
  {
    jastrow_timers[i]->start();
    std::vector<valT> ratios_jas(numWalkers*numKnots);
    std::vector<WaveFunctionComponent*> jas_list;
    for (int j = 0; j < packedIndex.size(); j++) {
      jas_list.push_back(WF_list[packedIndex[j]]->Jastrows[i]);
    }
    //std::cout << "in multi_ratio, about to call Jastrow[i]->multi_evalRatio for i = " << i << std::endl;
    Jastrows[i]->multi_evalRatio(pairNum, eiList, jas_list, apsd, likeTempRs, unlikeTempRs, activeWalkerIndex, ratios_jas); // handing in both because we don't know what type each is...

    for (int idx = 0; idx < numWalkers*numKnots; idx++)
      ratios[idx] *= ratios_jas[idx];
    jastrow_timers[i]->stop();
  }
}
*/

template<typename apsdType, typename psiVType, typename likeTempRType, typename unlikeTempRType, typename eiListType>
void WaveFunction::multi_ratio(int pairNum, WaveFunctionKokkos& wfc, apsdType& apsd, psiVType& tempPsiV,
			       likeTempRType& likeTempRs, unlikeTempRType& unlikeTempRs, eiListType& eiList,
			       std::vector<valT>& ratios) {
  int numWalkers = wfc.upDets.extent(0);
  int numKnots = tempPsiV.extent(1);
  std::vector<valT> ratios_det(numWalkers*numKnots, 1);
  for (int iw = 0; iw < numWalkers*numKnots; iw++) {
    ratios[iw] = valT(1.0);
  }

  auto tmpEiList = eiList;
  int numActive = 0;

  auto upDets = wfc.upDets;
  auto downDets = wfc.downDets;
  auto activeDDs = wfc.activeDDs;
  auto isActive = wfc.isActive;
  auto activeMap = wfc.activeMap;

  int locNelUp = this->nelup;

  Kokkos::parallel_reduce("set-up-worklist", Kokkos::RangePolicy<>(0, numWalkers),
			  KOKKOS_LAMBDA(const int& i, int &locActive) {			    
			    const int elNum = tmpEiList(i,pairNum,0);
			    if (elNum >= 0) {
			      isActive(i) = 1;
			      if (elNum < locNelUp) {
				activeDDs(i) = upDets(i);
				activeDDs(i).FirstIndex(0) = upDets(i).FirstIndex(0);
			      } else {
				activeDDs(i) = downDets(i);
				activeDDs(i).FirstIndex(0) = downDets(i).FirstIndex(0);
			      }
			      locActive++;
			    } else {
			      isActive(i) = 0;
			    }
			  }, numActive);
			   
  if (numActive > 0) {
    timers[Timer_Det]->start();
    Kokkos::parallel_for("set-up-map", Kokkos::RangePolicy<>(0, 1),
			 KOKKOS_LAMBDA(const int& i) {
			   int idx = 0;
			   for (int j = 0; j < upDets.extent(0); j++) {
			     if (isActive(j) == 1) {
			       activeMap(idx) = j;
			       idx++;
			     }
			   }
			 });
    Kokkos::deep_copy(wfc.activeMapMirror, wfc.activeMap);

    doDiracDeterminantMultiEvalRatio(pairNum, wfc, eiList, tempPsiV, ratios_det, numActive);

    for (int i = 0; i < numWalkers*numKnots; i++) {
      ratios[i] = ratios_det[i];
    }
    timers[Timer_Det]->stop();

    for (size_t i = 0; i < Jastrows.size(); i++)
    {
      jastrow_timers[i]->start();
      std::vector<valT> ratios_jas(numWalkers*numKnots, 1);
      //std::cout << "in multi_ratio, about to call Jastrow[i]->multi_evalRatio for i = " << i << std::endl;
      Jastrows[i]->multi_evalRatio(pairNum, eiList, wfc, apsd, likeTempRs, unlikeTempRs, ratios_jas, numActive); // handing in both because we don't know what type each is...

      for (int idx = 0; idx < numWalkers*numKnots; idx++)
	ratios[idx] *= ratios_jas[idx];
      jastrow_timers[i]->stop();
    }
  }
}


template<typename valsType, typename isValidMapMirrorType >
void WaveFunction::multi_ratioGrad(const std::vector<WaveFunction*>& WF_list,
				   WaveFunctionKokkos wfc,
				   Kokkos::View<ParticleSet::pskType*>& psk,
				   valsType& psiVs,
				   Kokkos::View<int*>& isValidMap, 
				   isValidMapMirrorType& isValidMapMirror, int numValid,
				   int iel, std::vector<valT>& ratio_list,
				   std::vector<posT>& grad_new) const {
  if (numValid > 0) {
    timers[Timer_Det]->start();
    std::vector<valT> ratios_det(numValid);
    for (int iw = 0; iw < grad_new.size(); iw++)
      grad_new[iw] = posT(0);
    
    Kokkos::View<valT*> tempResults("tempResults", numValid);
    if (iel < nelup) {
      //std::cout << "in multi_ratioGrad, using up determinant, iel = " << iel << std::endl;
      std::vector<WaveFunctionComponent*> up_list(extract_up_list(WF_list));
      //Det_up->multi_ratioGrad(up_list, P_list, iat, ratios_det, grad_new);
      doDiracDeterminantMultiEvalRatio(wfc.upDets, up_list, psiVs, tempResults, isValidMap, numValid, iel);
    } else {
      //std::cout << "in multi_ratioGrad, using down determinant, iel = " << iel << std::endl;
      std::vector<WaveFunctionComponent*> dn_list(extract_dn_list(WF_list));
      //Det_up->multi_ratioGrad(up_list, P_list, iat, ratios_det, grad_new);
      doDiracDeterminantMultiEvalRatio(wfc.downDets, dn_list, psiVs, tempResults, isValidMap, numValid, iel);
    }
    auto tempResultsMirror = Kokkos::create_mirror_view(tempResults);
    Kokkos::deep_copy(tempResultsMirror, tempResults);
    for (int i = 0; i < ratios_det.size(); i++) {
      ratios_det[i] = tempResultsMirror(i);
    }
    for (int iw = 0; iw < numValid; iw++) {
      ratio_list[isValidMapMirror(iw)] = ratios_det[iw];
    }
    timers[Timer_Det]->stop();

    for (size_t i = 0; i < Jastrows.size(); i++)
    {
      //std::cout << "    doing multi_ratioGrad for Jastrow " << i << std::endl;
      jastrow_timers[i]->start();
      std::vector<valT> ratios_jas(numValid);
      std::vector<WaveFunctionComponent*> jas_list(extract_jas_list(WF_list, i));
      //Jastrows[i]->multi_ratioGrad(jas_list, P_list, iat, ratios_jas, grad_new);
      Jastrows[i]->multi_ratioGrad(jas_list, wfc, psk, iel, isValidMap, numValid, ratios_jas, grad_new);
      for (int iw = 0; iw < numValid; iw++)
	ratio_list[isValidMapMirror(iw)] *= ratios_jas[iw];
      jastrow_timers[i]->stop();
    }
  }
}

 
} // namespace qmcplusplus

#endif
