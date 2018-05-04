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
 * @file WaveFunction.cpp
   @brief Wavefunction based on Structure of Arrays (SoA) storage
 */

namespace qmcplusplus
{

void build_WaveFunction(bool useRef, WaveFunction &WF, ParticleSet &ions, ParticleSet &els, const RandomGenerator<QMCTraits::RealType> &RNG, bool enableJ3)
{
  using valT = WaveFunction::valT;
  using posT = WaveFunction::posT;

  if(WF.Is_built)
  {
    app_log() << "The wavefunction was built before!" << std::endl;
    return;
  }

  const int nelup = els.getTotalNum()/2;

  if(useRef)
  {
  }
  else
  {
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
    WF.nelup = nelup;
    WF.Det_up = new DetType(nelup, RNG, 0);
    WF.Det_dn = new DetType(els.getTotalNum()-nelup, RNG, nelup);

    // J1 component
    J1OrbType *J1 = new J1OrbType(ions, els);
    buildJ1(*J1, els.Lattice.WignerSeitzRadius);
    WF.Jastrows.push_back(J1);

    // J2 component
    J2OrbType *J2 = new J2OrbType(els);
    buildJ2(*J2, els.Lattice.WignerSeitzRadius);
    WF.Jastrows.push_back(J2);

    // J3 component
    if(enableJ3)
    {
      J3OrbType *J3 = new J3OrbType(ions, els);
      buildJeeI(*J3, els.Lattice.WignerSeitzRadius);
      WF.Jastrows.push_back(J3);
    }
  }

  WF.Is_built = true;
}

WaveFunction::~WaveFunction()
{
  if (Is_built)
  {
    delete Det_up;
    delete Det_dn;
    for(size_t i=0; i<Jastrows.size(); i++)
      delete Jastrows[i];
  }
}

void WaveFunction::evaluateLog(ParticleSet &P)
{
  constexpr valT czero(0);
  if (FirstTime)
  {
    P.G       = czero;
    P.L       = czero;
    LogValue  = Det_up -> evaluateLog(P, P.G, P.L);
    LogValue += Det_dn -> evaluateLog(P, P.G, P.L);
    for(size_t i=0; i<Jastrows.size(); i++)
      LogValue += Jastrows[i]->evaluateLog(P, P.G, P.L);
    FirstTime = false;
  }
}

WaveFunctionBase::posT WaveFunction::evalGrad(ParticleSet &P, int iat)
{
  posT grad_iat = ( iat<nelup ? Det_up->evalGrad(P, iat) : Det_dn->evalGrad(P, iat) );
  for(size_t i=0; i<Jastrows.size(); i++)
    grad_iat += Jastrows[i]->evalGrad(P, iat);
  return grad_iat;
}

WaveFunctionBase::valT WaveFunction::ratioGrad(ParticleSet &P, int iat,
                                               posT &grad)
{
  valT ratio = ( iat<nelup ? Det_up->ratioGrad(P, iat, grad) : Det_dn->ratioGrad(P, iat, grad) );
  for(size_t i=0; i<Jastrows.size(); i++)
    ratio *= Jastrows[i]->ratioGrad(P, iat, grad);
  return ratio;
}

WaveFunctionBase::valT WaveFunction::ratio(ParticleSet &P, int iat)
{
  valT ratio = ( iat<nelup ? Det_up->ratio(P, iat) : Det_dn->ratio(P, iat) );
  for(size_t i=0; i<Jastrows.size(); i++)
    ratio *= Jastrows[i]->ratio(P, iat);
  return ratio;
}

void WaveFunction::acceptMove(ParticleSet &P, int iat)
{
  if(iat<nelup)
    Det_up->acceptMove(P, iat);
  else
    Det_dn->acceptMove(P, iat);
  for(size_t i=0; i<Jastrows.size(); i++)
    Jastrows[i]->acceptMove(P, iat);
}

void WaveFunction::restore(int iat) {}

void WaveFunction::evaluateGL(ParticleSet &P)
{
  constexpr valT czero(0);
  P.G = czero;
  P.L = czero;
  Det_up->evaluateGL(P, P.G, P.L);
  Det_dn->evaluateGL(P, P.G, P.L);
  for(size_t i=0; i<Jastrows.size(); i++)
    Jastrows[i]->evaluateGL(P, P.G, P.L);
}
} // qmcplusplus
