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

#ifndef QMCPLUSPLUS_WAVEFUNCTION_H
#define QMCPLUSPLUS_WAVEFUNCTION_H
#include <Utilities/Configuration.h>
#include "Devices.h"

#include <Utilities/RandomGenerator.h>
#include <Utilities/NewTimer.h>
#include <Particle/ParticleSet.h>
#include <QMCWaveFunctions/WaveFunctionComponent.h>

namespace qmcplusplus
{
enum WaveFunctionTimers
{
  Timer_Det,
  Timer_Finish,
  Timer_GL,
};

template<Devices>
class WaveFunctionBuilder;
/** A minimal TrialWavefunction
 */
class WaveFunction
{
  TimerNameLevelList_t<WaveFunctionTimers> WaveFunctionTimerNames;

public:
  using RealType = OHMMS_PRECISION;
  using valT     = OHMMS_PRECISION;
  using posT     = TinyVector<valT, OHMMS_DIM>;

protected:
  /** this shouldn't be pointers and it should be able to express that some Jastrows can be called in parallel. */
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
      : FirstTime(true), Is_built(false), nelup(0), ei_TableID(1), Det_up(nullptr), Det_dn(nullptr), LogValue(0.0)
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
  void multi_evaluateLog(const std::vector<WaveFunction*>& WF_list, const std::vector<ParticleSet*>& P_list) const;
  void multi_evalGrad(const std::vector<WaveFunction*>& WF_list,
                      const std::vector<ParticleSet*>& P_list,
                      int iat,
                      std::vector<posT>& grad_now) const;
  void multi_ratioGrad(const std::vector<WaveFunction*>& WF_list,
                       const std::vector<ParticleSet*>& P_list,
                       int iat,
                       std::vector<valT>& ratio_list,
                       std::vector<posT>& grad_new) const;
  void multi_ratio(const std::vector<ParticleSet*>& P_list, int iat) const {};
  void multi_acceptrestoreMove(const std::vector<WaveFunction*>& WF_list,
                               const std::vector<ParticleSet*>& P_list,
                               const std::vector<bool>& isAccepted,
                               int iat) const;
  void multi_evaluateGL(const std::vector<WaveFunction*>& WF_list, const std::vector<ParticleSet*>& P_list) const;

  // others
  void finishUpdates(int iel)
  {
    assert(Det_up != nullptr);
    assert(Det_up != nullptr);
    timers[Timer_Finish]->start();
    if (iel < nelup)
      Det_up->finishUpdate(iel);
    else
      Det_dn->finishUpdate(iel);
    timers[Timer_Finish]->stop();
  }
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

  template<Devices D>
  friend class WaveFunctionBuilder;
};


} // namespace qmcplusplus

#endif
