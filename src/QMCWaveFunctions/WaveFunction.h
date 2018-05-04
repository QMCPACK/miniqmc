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
 * WaveFunctionComponentBase).
 *
 * Corresponds to QMCWaveFunction/TrialWaveFunction.h in the QMCPACK source.
 */

#ifndef QMCPLUSPLUS_WAVEFUNCTIONS_H
#define QMCPLUSPLUS_WAVEFUNCTIONS_H
#include <Utilities/Configuration.h>
#include <Utilities/RandomGenerator.h>
#include <Particle/DistanceTable.h>
#include <QMCWaveFunctions/Determinant.h>
#include <QMCWaveFunctions/DeterminantRef.h>
#include <QMCWaveFunctions/Jastrow/BsplineFunctor.h>
#include <QMCWaveFunctions/Jastrow/PolynomialFunctor3D.h>
#include <QMCWaveFunctions/Jastrow/OneBodyJastrowRef.h>
#include <QMCWaveFunctions/Jastrow/OneBodyJastrow.h>
#include <QMCWaveFunctions/Jastrow/TwoBodyJastrowRef.h>
#include <QMCWaveFunctions/Jastrow/TwoBodyJastrow.h>
#include <QMCWaveFunctions/Jastrow/ThreeBodyJastrowRef.h>
#include <QMCWaveFunctions/Jastrow/ThreeBodyJastrow.h>

namespace qmcplusplus
{
/** A minimal TrialWavefunction
 */

struct WaveFunction
{
  friend void build_WaveFunction(bool useRef, WaveFunction &WF, ParticleSet &ions, ParticleSet &els, const RandomGenerator<QMCTraits::RealType> &RNG, bool enableJ3);

  using RealType = OHMMS_PRECISION;
  using valT = OHMMS_PRECISION;
  using posT = TinyVector<valT, OHMMS_DIM>;

  private:
  std::vector<WaveFunctionComponentBase *> Jastrows;
  WaveFunctionComponentBase *Det_up;
  WaveFunctionComponentBase *Det_dn;
  valT LogValue;

  bool FirstTime, Is_built;
  int nelup, ei_TableID;

  public:
  WaveFunction(): FirstTime(true), Is_built(false), nelup(0), ei_TableID(1), Det_up(nullptr), Det_dn(nullptr) { }
  ~WaveFunction();
  void evaluateLog(ParticleSet &P);
  posT evalGrad(ParticleSet &P, int iat);
  valT ratioGrad(ParticleSet &P, int iat, posT &grad);
  valT ratio(ParticleSet &P, int iat);
  void acceptMove(ParticleSet &P, int iat);
  void restore(int iat);
  void evaluateGL(ParticleSet &P);
  int get_ei_TableID() const {return ei_TableID;}
};

void build_WaveFunction(bool useRef, WaveFunction &WF, ParticleSet &ions, ParticleSet &els, const RandomGenerator<QMCTraits::RealType> &RNG, bool enableJ3);

} // qmcplusplus

#endif
