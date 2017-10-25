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
#include <omp.h>
#include <QMCWaveFunctions/FakeWaveFunction.h>
#include <Input/Input.hpp>

/*!
 * @file SoAWaveFunction.cpp
   @brief Wavefunction based on Structure of Arrays (SoA) storage
 */

namespace qmcplusplus
{
WaveFunction::WaveFunction(ParticleSet &ions, ParticleSet &els)
{
  FirstTime = true;

  ions.RSoA = ions.R;
  els.RSoA  = els.R;

  d_ee = DistanceTable::add(els, DT_SOA);
  d_ie = DistanceTable::add(ions, els, DT_SOA);

  int ip = omp_get_thread_num();
  J2     = new J2OrbType(els);
  buildJ2(*J2, els.Lattice.WignerSeitzRadius);
}

WaveFunction::~WaveFunction() { delete J2; }

void WaveFunction::evaluateLog(ParticleSet &P)
{
  constexpr valT czero(0);
  if (FirstTime)
  {
    P.G       = czero;
    P.L       = czero;
    LogValue  = J2->evaluateLog(P, P.G, P.L);
    FirstTime = false;
  }
}

FakeWaveFunctionBase::posT WaveFunction::evalGrad(ParticleSet &P, int iat)
{
  return J2->evalGrad(P, iat);
}

FakeWaveFunctionBase::valT WaveFunction::ratioGrad(ParticleSet &P, int iat,
                                                      posT &grad)
{
  return J2->ratioGrad(P, iat, grad);
}

FakeWaveFunctionBase::valT WaveFunction::ratio(ParticleSet &P, int iat)
{
  return J2->ratio(P, iat);
}
void WaveFunction::acceptMove(ParticleSet &P, int iat)
{
  J2->acceptMove(P, iat);
}
void WaveFunction::restore(int iat) {}

void WaveFunction::evaluateGL(ParticleSet &P)
{
  constexpr valT czero(0);
  P.G = czero;
  P.L = czero;
  J2->evaluateGL(P, P.G, P.L);
}
}
