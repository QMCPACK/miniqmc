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
#include <QMCWaveFunctions/WaveFunction.h>
#include <Input/Input.hpp>

/*!
 * @file SoAWaveFunction.cpp
   @brief Wavefunction based on Structure of Arrays (SoA) storage
 */

namespace qmcplusplus
{
WaveFunction::WaveFunction(ParticleSet &ions, ParticleSet &els,
                           RandomGenerator<RealType> RNG)
{
  FirstTime = true;

  ions.RSoA = ions.R;
  els.RSoA  = els.R;

  // distance tables
  d_ee = DistanceTable::add(els, DT_SOA);
  d_ie = DistanceTable::add(ions, els, DT_SOA);

  // determinant component
  Det = new DetType(els.getTotalNum(), RNG);

  // J1 component
  J1 = new J1OrbType(ions, els);
  buildJ1(*J1, els.Lattice.WignerSeitzRadius);

  // J2 component
  J2 = new J2OrbType(els);
  buildJ2(*J2, els.Lattice.WignerSeitzRadius);

  // J3 component
  J3 = new J3OrbType(ions, els);
  buildJeeI(*J3, els.Lattice.WignerSeitzRadius);
}

WaveFunction::~WaveFunction() { delete J2; }

void WaveFunction::evaluateLog(ParticleSet &P)
{
  constexpr valT czero(0);
  if (FirstTime)
  {
    P.G       = czero;
    P.L       = czero;
    LogValue  = Det->evaluateLog(P, P.G, P.L);
    LogValue += J1->evaluateLog(P, P.G, P.L);
    LogValue += J2->evaluateLog(P, P.G, P.L);
    LogValue += J3->evaluateLog(P, P.G, P.L);
    FirstTime = false;
  }
}

WaveFunctionBase::posT WaveFunction::evalGrad(ParticleSet &P, int iat)
{
  return Det->evalGrad(P, iat) + J1->evalGrad(P, iat) + J2->evalGrad(P, iat) +
         J3->evalGrad(P, iat);
}

WaveFunctionBase::valT WaveFunction::ratioGrad(ParticleSet &P, int iat,
                                               posT &grad)
{
  return Det->ratioGrad(P, iat, grad) + J1->ratioGrad(P, iat, grad) +
         J2->ratioGrad(P, iat, grad) + J3->ratioGrad(P, iat, grad);
}

WaveFunctionBase::valT WaveFunction::ratio(ParticleSet &P, int iat)
{
  return Det->ratio(P, iat) * J1->ratio(P, iat) * J2->ratio(P, iat) *
         J3->ratio(P, iat);
}

void WaveFunction::acceptMove(ParticleSet &P, int iat)
{
  Det->acceptMove(P, iat);
  J1->acceptMove(P, iat);
  J2->acceptMove(P, iat);
  J3->acceptMove(P, iat);
}

void WaveFunction::restore(int iat) {}

void WaveFunction::evaluateGL(ParticleSet &P)
{
  constexpr valT czero(0);
  P.G = czero;
  P.L = czero;
  Det->evaluateGL(P, P.G, P.L);
  J1->evaluateGL(P, P.G, P.L);
  J2->evaluateGL(P, P.G, P.L);
  J3->evaluateGL(P, P.G, P.L);
}
} // qmcplusplus
