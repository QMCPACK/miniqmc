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
#include <Particle/VirtualParticleSet.h>
#include <QMCWaveFunctions/SPOSet_builder.h>
#include <QMCWaveFunctions/WaveFunctionComponent.h>

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
  /// Slater determinants
  WaveFunctionComponent* Det_up;
  WaveFunctionComponent* Det_dn;
  /// Jastrow factors
  std::vector<WaveFunctionComponent*> Jastrows;
  valT LogValue;

  bool FirstTime, Is_built;
  int nelup, ei_TableID;

  TimerList_t timers;
  TimerList_t jastrow_timers;

public:
  WaveFunction();
  ~WaveFunction();

  /// operates on a single walker
  void evaluateLog(ParticleSet& P);
  posT evalGrad(ParticleSet& P, int iat);
  valT ratioGrad(ParticleSet& P, int iat, posT& grad);
  valT ratio(ParticleSet& P, int iat);
  void acceptMove(ParticleSet& P, int iat);
  void restore(int iat);
  void completeUpdates();
  void evaluateGL(ParticleSet& P);

  /** compulte multiple ratios to handle non-local moves and other virtual moves
   */
  void evaluateRatios(VirtualParticleSet& P, std::vector<valT>& ratios);

  /// operates on multiple walkers
  void flex_evaluateLog(const std::vector<WaveFunction*>& WF_list,
                         const std::vector<ParticleSet*>& P_list) const;
  void flex_evalGrad(const std::vector<WaveFunction*>& WF_list,
                      const std::vector<ParticleSet*>& P_list,
                      int iat,
                      std::vector<posT>& grad_now) const;
  void flex_ratioGrad(const std::vector<WaveFunction*>& WF_list,
                       const std::vector<ParticleSet*>& P_list,
                       int iat,
                       std::vector<valT>& ratio_list,
                       std::vector<posT>& grad_new) const;
  void flex_ratio(const std::vector<ParticleSet*>& P_list, int iat) const {};
  void flex_acceptrestoreMove(const std::vector<WaveFunction*>& WF_list,
                               const std::vector<ParticleSet*>& P_list,
                               const std::vector<bool>& isAccepted,
                               int iat) const;
  void flex_evaluateGL(const std::vector<WaveFunction*>& WF_list,
                        const std::vector<ParticleSet*>& P_list) const;

  void flex_completeUpdates(const std::vector<WaveFunction*>& WF_list) const;

  // others
  int get_ei_TableID() const { return ei_TableID; }
  valT getLogValue() const { return LogValue; }
  void setupTimers();

  // friends
  friend void build_WaveFunction(bool useRef,
                                 const SPOSet* spo_main,
                                 WaveFunction& WF,
                                 ParticleSet& ions,
                                 ParticleSet& els,
                                 const RandomGenerator<QMCTraits::RealType>& RNG,
                                 int delay_rank,
                                 bool enableJ3);
  const std::vector<WaveFunctionComponent*>
      extract_up_list(const std::vector<WaveFunction*>& WF_list) const;
  const std::vector<WaveFunctionComponent*>
      extract_dn_list(const std::vector<WaveFunction*>& WF_list) const;
  const std::vector<WaveFunctionComponent*>
      extract_jas_list(const std::vector<WaveFunction*>& WF_list, int jas_id) const;
};

void build_WaveFunction(bool useRef,
                        const SPOSet* spo_main,
                        WaveFunction& WF,
                        ParticleSet& ions,
                        ParticleSet& els,
                        const RandomGenerator<QMCTraits::RealType>& RNG,
                        int delay_rank,
                        bool enableJ3);
} // namespace qmcplusplus

#endif
