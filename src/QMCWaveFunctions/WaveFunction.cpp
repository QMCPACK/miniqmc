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
 * @file WaveFunction.cpp
 * @brief Wavefunction based on Structure of Arrays (SoA) storage
 */

#include <QMCWaveFunctions/WaveFunction.h>
#include <QMCWaveFunctions/Determinant.h>
#include <QMCWaveFunctions/DeterminantDevice.h>
#include <QMCWaveFunctions/DeterminantDeviceImp.h>
#include <QMCWaveFunctions/DeterminantRef.h>
#include <QMCWaveFunctions/Jastrow/BsplineFunctor.h>
#include <QMCWaveFunctions/Jastrow/BsplineFunctorRef.h>
#include <QMCWaveFunctions/Jastrow/PolynomialFunctor3D.h>
#include <QMCWaveFunctions/Jastrow/OneBodyJastrowRef.h>
#include <QMCWaveFunctions/Jastrow/OneBodyJastrow.h>
#include <QMCWaveFunctions/Jastrow/TwoBodyJastrowRef.h>
#include <QMCWaveFunctions/Jastrow/TwoBodyJastrow.h>
#include <QMCWaveFunctions/Jastrow/ThreeBodyJastrowRef.h>
#include <QMCWaveFunctions/Jastrow/ThreeBodyJastrow.h>
#include <Input/Input.hpp>

namespace qmcplusplus
{
WaveFunction::~WaveFunction()
{
  if (Is_built)
  {
    delete Det_up;
    delete Det_dn;
    for (size_t i = 0; i < Jastrows.size(); i++)
      delete Jastrows[i];
  }
}

void WaveFunction::setupTimers()
{
  TimerNameLevelList_t<WaveFunctionTimers> WaveFunctionTimerNames = {{Timer_Det, "Determinant", timer_level_fine},
                                                                     {Timer_Finish, "FinishUpdate", timer_level_fine},
                                                                     {Timer_GL, "Kinetic Energy", timer_level_coarse}};

  setup_timers(timers, WaveFunctionTimerNames);
  for (int i = 0; i < Jastrows.size(); i++)
  {
    jastrow_timers.push_back(
        TimerManagerClass::get().createTimer(Jastrows[i]->WaveFunctionComponentName, timer_level_fine));
  }
}

void WaveFunction::evaluateLog(ParticleSet& P)
{
  constexpr valT czero(0);
  if (FirstTime)
  {
    P.G      = czero;
    P.L      = czero;
    LogValue = Det_up->evaluateLog(P, P.G, P.L);
    LogValue += Det_dn->evaluateLog(P, P.G, P.L);
    for (size_t i = 0; i < Jastrows.size(); i++)
      LogValue += Jastrows[i]->evaluateLog(P, P.G, P.L);
    FirstTime = false;
  }
}

WaveFunction::posT WaveFunction::evalGrad(ParticleSet& P, int iat)
{
  timers[Timer_Det]->start();
  posT grad_iat = (iat < nelup ? Det_up->evalGrad(P, iat) : Det_dn->evalGrad(P, iat));
  timers[Timer_Det]->stop();

  for (size_t i = 0; i < Jastrows.size(); i++)
  {
    jastrow_timers[i]->start();
    grad_iat += Jastrows[i]->evalGrad(P, iat);
    jastrow_timers[i]->stop();
  }
  return grad_iat;
}

WaveFunction::valT WaveFunction::ratioGrad(ParticleSet& P, int iat, posT& grad)
{
  timers[Timer_Det]->start();
  grad       = valT(0);
  valT ratio = (iat < nelup ? Det_up->ratioGrad(P, iat, grad) : Det_dn->ratioGrad(P, iat, grad));
  timers[Timer_Det]->stop();

  for (size_t i = 0; i < Jastrows.size(); i++)
  {
    jastrow_timers[i]->start();
    ratio *= Jastrows[i]->ratioGrad(P, iat, grad);
    jastrow_timers[i]->stop();
  }
  return ratio;
}

WaveFunction::valT WaveFunction::ratio(ParticleSet& P, int iat)
{
  timers[Timer_Det]->start();
  valT ratio = (iat < nelup ? Det_up->ratio(P, iat) : Det_dn->ratio(P, iat));
  timers[Timer_Det]->stop();

  for (size_t i = 0; i < Jastrows.size(); i++)
  {
    jastrow_timers[i]->start();
    ratio *= Jastrows[i]->ratio(P, iat);
    jastrow_timers[i]->stop();
  }
  return ratio;
}

void WaveFunction::acceptMove(ParticleSet& P, int iat)
{
  timers[Timer_Det]->start();
  if (iat < nelup)
    Det_up->acceptMove(P, iat);
  else
    Det_dn->acceptMove(P, iat);
  timers[Timer_Det]->stop();

  for (size_t i = 0; i < Jastrows.size(); i++)
  {
    jastrow_timers[i]->start();
    Jastrows[i]->acceptMove(P, iat);
    jastrow_timers[i]->stop();
  }
}

void WaveFunction::restore(int iat) {}

void WaveFunction::evaluateGL(ParticleSet& P)
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

  for (size_t i = 0; i < Jastrows.size(); i++)
  {
    jastrow_timers[i]->start();
    Jastrows[i]->evaluateGL(P, P.G, P.L);
    LogValue += Jastrows[i]->LogValue;
    jastrow_timers[i]->stop();
  }
}


} // namespace qmcplusplus
