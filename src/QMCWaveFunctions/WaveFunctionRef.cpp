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
#include <QMCWaveFunctions/WaveFunction.h>
#include <Input/Input.hpp>

/*!
 * @file WaveFunctionRef.cpp
 * @brief Wavefunction based on reference implemenation
 */

/*!
 *  Namespace containing reference implementation.
 */

namespace miniqmcreference
{
using namespace qmcplusplus;

WaveFunctionRef::WaveFunctionRef(ParticleSet &ions, ParticleSet &els,
                                 RandomGenerator<RealType> RNG, bool enableJ3)
{
  FirstTime = true;

  ions.RSoA = ions.R;
  els.RSoA  = els.R;

  // distance tables
  d_ee = DistanceTable::add(els, DT_SOA);
  d_ie = DistanceTable::add(ions, els, DT_SOA);

  // determinant component
  nelup = els.getTotalNum()/2;
  Det_up = new DetType(nelup, RNG, 0);
  Det_dn = new DetType(els.getTotalNum()-nelup, RNG, nelup);

  // J1 component
  J1 = new J1OrbType(ions, els);
  buildJ1(*J1, els.Lattice.WignerSeitzRadius);

  // J2 component
  J2 = new J2OrbType(els);
  buildJ2(*J2, els.Lattice.WignerSeitzRadius);

  // J3 component
  if(enableJ3)
  {
    J3 = new J3OrbType(ions, els);
    buildJeeI(*J3, els.Lattice.WignerSeitzRadius);
  }
  else
    J3 = nullptr;
}

WaveFunctionRef::~WaveFunctionRef()
{
  delete Det_up;
  delete Det_dn;
  delete J1;
  delete J2;
  if(J3) delete J3;
}

void WaveFunctionRef::evaluateLog(ParticleSet &P)
{
  constexpr valT czero(0);
  if (FirstTime)
  {
    P.G       = czero;
    P.L       = czero;
    LogValue  = Det_up->evaluateLog(P, P.G, P.L);
    LogValue += Det_dn->evaluateLog(P, P.G, P.L);
    LogValue += J1->evaluateLog(P, P.G, P.L);
    LogValue += J2->evaluateLog(P, P.G, P.L);
    if(J3) LogValue += J3->evaluateLog(P, P.G, P.L);
    FirstTime = false;
  }
}

WaveFunctionBase::posT WaveFunctionRef::evalGrad(ParticleSet &P, int iat)
{
  return ( iat<nelup ? Det_up->evalGrad(P, iat) : Det_dn->evalGrad(P, iat) )
         + J1->evalGrad(P, iat)
         + J2->evalGrad(P, iat)
         + (J3?J3->evalGrad(P, iat):posT());
}

WaveFunctionBase::valT WaveFunctionRef::ratioGrad(ParticleSet &P, int iat,
                                                  posT &grad)
{
  return ( iat<nelup ? Det_up->ratioGrad(P, iat, grad) : Det_dn->ratioGrad(P, iat, grad) )
         * J1->ratioGrad(P, iat, grad)
         * J2->ratioGrad(P, iat, grad)
         * (J3?J3->ratioGrad(P, iat, grad):1.0);
}

WaveFunctionBase::valT WaveFunctionRef::ratio(ParticleSet &P, int iat)
{
  return ( iat<nelup ? Det_up->ratio(P, iat) : Det_dn->ratio(P, iat) )
         * J1->ratio(P, iat)
         * J2->ratio(P, iat)
         * (J3?J3->ratio(P, iat):1.0);
}

void WaveFunctionRef::acceptMove(ParticleSet &P, int iat)
{
  if(iat<nelup)
    Det_up->acceptMove(P, iat);
  else
    Det_dn->acceptMove(P, iat);
  J1->acceptMove(P, iat);
  J2->acceptMove(P, iat);
  if(J3) J3->acceptMove(P, iat);
}

void WaveFunctionRef::restore(int iat) {}

void WaveFunctionRef::evaluateGL(ParticleSet &P)
{
  constexpr valT czero(0);
  P.G = czero;
  P.L = czero;
  Det_up->evaluateGL(P, P.G, P.L);
  Det_dn->evaluateGL(P, P.G, P.L);
  J1->evaluateGL(P, P.G, P.L);
  J2->evaluateGL(P, P.G, P.L);
  if(J3) J3->evaluateGL(P, P.G, P.L);
}
} // miniqmcreferencce
