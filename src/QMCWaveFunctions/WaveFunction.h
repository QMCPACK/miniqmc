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
			 WaveFunctionKokkos& wfc,
			 Kokkos::View<ParticleSet::pskType*>& psk) const;

  void multi_evalGrad(const std::vector<WaveFunction*>& WF_list,
                      WaveFunctionKokkos& wfc,
		      Kokkos::View<ParticleSet::pskType*>& psk,
                      int iat,
                      std::vector<posT>& grad_now) const;

  // think about making this not a template!!!
  template<typename valsType, typename isValidMapMirrorType>
  void multi_ratioGrad(const std::vector<WaveFunction*>& WF_list,
		       WaveFunctionKokkos wfc,
		       Kokkos::View<ParticleSet::pskType*>& psk,
		       valsType& psiVs,
		       Kokkos::View<int*>& isValidMap, 
		       isValidMapMirrorType& isValidMapMirror, int numValid,
		       int iel, std::vector<valT>& ratio_list,
		       std::vector<posT>& grad_new) const;


  template<typename apsdType, typename psiVType, typename likeTempRType, 
           typename unlikeTempRType, typename eiListType>
  void multi_ratio(int pairNum, WaveFunctionKokkos& wfc, apsdType& apsd, psiVType& tempPsiV,
		   likeTempRType& likeTempRs, unlikeTempRType& unlikeTempRs, eiListType& eiList, 
		   std::vector<valT>& ratios); 

  void multi_acceptrestoreMove(const std::vector<WaveFunction*>& WF_list,
			       WaveFunctionKokkos& wfc,
			       Kokkos::View<ParticleSet::pskType*>& psk,
			       Kokkos::View<int*>& isAcceptedMap,
			       int numAccepted, int iel) const;

  void multi_evaluateGL(WaveFunctionKokkos& wfc,
			Kokkos::View<ParticleSet::pskType*>& apsk) const;


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



template<typename apsdType, typename psiVType, typename likeTempRType, 
         typename unlikeTempRType, typename eiListType>
void WaveFunction::multi_ratio(int pairNum, WaveFunctionKokkos& wfc, apsdType& apsd, psiVType& tempPsiV,
			       likeTempRType& likeTempRs, unlikeTempRType& unlikeTempRs, eiListType& eiList,
			       std::vector<valT>& ratios) {
  int numWalkers = wfc.upDets.extent(0);
  int numKnots = tempPsiV.extent(1);
  std::vector<valT> ratios_det(numWalkers*numKnots, 1);
  for (int iw = 0; iw < numWalkers*numKnots; iw++) {
    ratios[iw] = valT(1.0);
  }

  
  //std::cout << "numActive = " << numActive << std::endl;
  if (wfc.numActive > 0) {
    timers[Timer_Det]->start();    

    doDiracDeterminantMultiEvalRatio(pairNum, wfc, eiList, tempPsiV, ratios_det);

    for (int i = 0; i < numWalkers*numKnots; i++) {
      ratios[i] = ratios_det[i];
    }
    timers[Timer_Det]->stop();

    for (size_t i = 0; i < Jastrows.size(); i++)
    {
      jastrow_timers[i]->start();
      std::vector<valT> ratios_jas(numWalkers*numKnots, 1);
      Jastrows[i]->multi_evalRatio(pairNum, eiList, wfc, apsd, likeTempRs, unlikeTempRs, ratios_jas);

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
      doDiracDeterminantMultiEvalRatio(wfc.upDets, psiVs, tempResults, isValidMap, 
				       numValid, iel, wfc.numUpElectrons);
    } else {
      doDiracDeterminantMultiEvalRatio(wfc.downDets, psiVs, tempResults, isValidMap, 
				       numValid, iel, wfc.numDownElectrons);
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
      jastrow_timers[i]->start();
      std::vector<valT> ratios_jas(numValid);
      std::vector<WaveFunctionComponent*> jas_list(extract_jas_list(WF_list, i));
      Jastrows[i]->multi_ratioGrad(jas_list, wfc, psk, iel, isValidMap, numValid, ratios_jas, grad_new);
      for (int iw = 0; iw < numValid; iw++)
	ratio_list[isValidMapMirror(iw)] *= ratios_jas[iw];
      jastrow_timers[i]->stop();
    }
  }
}

 
} // namespace qmcplusplus

#endif
