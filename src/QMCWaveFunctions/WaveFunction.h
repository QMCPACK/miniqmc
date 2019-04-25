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

namespace qmcplusplus
{
/** A minimal TrialWavefunction
 */

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
  void evaluateLog(ParticleSet& P);
  posT evalGrad(ParticleSet& P, int iat);
  valT ratioGrad(ParticleSet& P, int iat, posT& grad);
  valT ratio(ParticleSet& P, int iat);
  void acceptMove(ParticleSet& P, int iat);
  void restore(int iat);
  void evaluateGL(ParticleSet& P);

  /// operates on multiple walkers
  void multi_evaluateLog(const std::vector<WaveFunction*>& WF_list,
                         const std::vector<ParticleSet*>& P_list) const;
  void multi_evalGrad(const std::vector<WaveFunction*>& WF_list,
                      const std::vector<ParticleSet*>& P_list,
                      int iat,
                      std::vector<posT>& grad_now) const;
  void multi_ratioGrad(const std::vector<WaveFunction*>& WF_list,
                       const std::vector<ParticleSet*>& P_list,
                       int iat,
                       std::vector<valT>& ratio_list,
                       std::vector<posT>& grad_new) const;

  template<typename psiVType, typename likeTempRType, typename unlikeTempRType, typename eiListType>
  void multi_ratio(int pairNum, const std::vector<WaveFunction*>& WF_list, psiVType& tempPsiV,
		   likeTempRType& likeTempRs, unlikeTempRType& unlikeTempRs, eiListType& eiList, std::vector<valT>& ratios); 

  void multi_acceptrestoreMove(const std::vector<WaveFunction*>& WF_list,
                               const std::vector<ParticleSet*>& P_list,
                               const std::vector<bool>& isAccepted,
                               int iat) const;
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

template<typename psiVType, typename likeTempRType, typename unlikeTempRType, typename eiListType>
void WaveFunction::multi_ratio(int pairNum, const std::vector<WaveFunction*>& WF_list, psiVType& tempPsiV,
			       likeTempRType& likeTempRs, unlikeTempRType& unlikeTempRs, eiListType& eiList,
			       std::vector<valT>& ratios) {
  // timers[Timer_Det]->start();
  int numWalkers = eiList.extent(0);
  int numKnots = tempPsiV.extent(1);
  std::vector<valT> ratios_det(numWalkers*numKnots);
  for (int iw = 0; iw < numWalkers*numKnots; iw++) {
    ratios[iw] = valT(0);
  }
  
  // complication here in that we are not always looking at the same electron from walker to walker
  auto EiListMirror = Kokkos::create_mirror_view(eiList);
  Kokkos::deep_copy(EiListMirror, eiList);

  std::vector<int> packedIndex;
  for (int iw = 0; iw < numWalkers; iw++) {
    const int elNum = EiListMirror(iw,pairNum,0);
    if (elNum > 0) {
      packedIndex.push_back(iw);
    }
  }
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
  doDiracDeterminantMultiEvalRatio(pairNum, addk, activeWalkerIndex, eiList, tempPsiV, ratios_det);

  for (int i = 0; i < numWalkers*numKnots; i++) {
    ratios[i] = ratios_det[i];
  }
				   //timers[Timer_Det]->stop();

  for (size_t i = 0; i < Jastrows.size(); i++)
  {
    jastrow_timers[i]->start();
    std::vector<valT> ratios_jas(numWalkers*numKnots);
    std::vector<WaveFunctionComponent*> jas_list;
    for (int j = 0; j < packedIndex.size(); j++) {
      jas_list.push_back(WF_list[packedIndex[j]]->Jastrows[i]);
    }
    Jastrows[i]->multi_evalRatio(pairNum, eiList, jas_list, likeTempRs, unlikeTempRs, activeWalkerIndex, ratios_jas); // handing in both because we don't know what type each is...

    for (int idx = 0; idx < numWalkers*numKnots; idx++)
      ratios[idx] *= ratios_jas[idx];
    jastrow_timers[i]->stop();
  }
}



 
} // namespace qmcplusplus

#endif
