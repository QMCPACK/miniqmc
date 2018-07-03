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

enum WaveFunctionTimers
{
  Timer_Det,
  Timer_GL,
};

TimerNameList_t<WaveFunctionTimers> WaveFunctionTimerNames = {
  {Timer_Det, "Determinant"},
  {Timer_GL, "Kinetic Energy"}
};


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
    using J1OrbType = miniqmcreference::OneBodyJastrowRef<BsplineFunctor<valT>>;
    using J2OrbType = miniqmcreference::TwoBodyJastrowRef<BsplineFunctor<valT>>;
    using J3OrbType = miniqmcreference::ThreeBodyJastrowRef<PolynomialFunctor3D>;
    using DetType   = miniqmcreference::DiracDeterminantRef;

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

  WF.setupTimers();

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

void WaveFunction::setupTimers()
{
  setup_timers(timers, WaveFunctionTimerNames, timer_level_coarse);
  for (int i = 0; i < Jastrows.size(); i++) {
    jastrow_timers.push_back(TimerManager.createTimer(Jastrows[i]->WaveFunctionComponentName, timer_level_coarse));
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

WaveFunction::posT WaveFunction::evalGrad(ParticleSet &P, int iat)
{
  timers[Timer_Det]->start();
  posT grad_iat = ( iat<nelup ? Det_up->evalGrad(P, iat) : Det_dn->evalGrad(P, iat) );
  timers[Timer_Det]->stop();

  for(size_t i=0; i<Jastrows.size(); i++) {
    jastrow_timers[i]->start();
    grad_iat += Jastrows[i]->evalGrad(P, iat);
    jastrow_timers[i]->stop();
  }
  return grad_iat;
}

WaveFunction::valT WaveFunction::ratioGrad(ParticleSet &P, int iat,
                                               posT &grad)
{
  timers[Timer_Det]->start();
  valT ratio = ( iat<nelup ? Det_up->ratioGrad(P, iat, grad) : Det_dn->ratioGrad(P, iat, grad) );
  timers[Timer_Det]->stop();

  for(size_t i=0; i<Jastrows.size(); i++) {
    jastrow_timers[i]->start();
    ratio *= Jastrows[i]->ratioGrad(P, iat, grad);
    jastrow_timers[i]->stop();
  }
  return ratio;
}

WaveFunction::valT WaveFunction::ratio(ParticleSet &P, int iat)
{

  timers[Timer_Det]->start();
  valT ratio = ( iat<nelup ? Det_up->ratio(P, iat) : Det_dn->ratio(P, iat) );
  timers[Timer_Det]->stop();

  for(size_t i=0; i<Jastrows.size(); i++)
  {
    jastrow_timers[i]->start();
    ratio *= Jastrows[i]->ratio(P, iat);
    jastrow_timers[i]->stop();
  }
  return ratio;
}

void WaveFunction::acceptMove(ParticleSet &P, int iat)
{
  timers[Timer_Det]->start();
  if(iat<nelup)
    Det_up->acceptMove(P, iat);
  else
    Det_dn->acceptMove(P, iat);
  timers[Timer_Det]->stop();

  for(size_t i=0; i<Jastrows.size(); i++)
  {
    jastrow_timers[i]->start();
    Jastrows[i]->acceptMove(P, iat);
    jastrow_timers[i]->stop();
  }
}

void WaveFunction::restore(int iat) {}

void WaveFunction::evaluateGL(ParticleSet &P)
{
  ScopedTimer local_timer(timers[Timer_GL]);

  constexpr valT czero(0);
  P.G = czero;
  P.L = czero;
  timers[Timer_Det]->start();
  Det_up->evaluateGL(P, P.G, P.L);
  Det_dn->evaluateGL(P, P.G, P.L);
  LogValue = Det_up->LogValue + Det_dn->LogValue;
  timers[Timer_Det]->stop();

  for(size_t i=0; i<Jastrows.size(); i++)
  {
    jastrow_timers[i]->start();
    Jastrows[i]->evaluateGL(P, P.G, P.L);
    LogValue += Jastrows[i]->LogValue;
    jastrow_timers[i]->stop();
  }
}
} // qmcplusplus
