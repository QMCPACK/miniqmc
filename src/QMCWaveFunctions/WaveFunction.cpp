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

/*!
 * @file WaveFunction.cpp
   @brief Wavefunction based on Structure of Arrays (SoA) storage
 */

#include <QMCWaveFunctions/WaveFunction.h>
#include <QMCWaveFunctions/DiracDeterminantRef.h>
#include <QMCWaveFunctions/DiracDeterminant.h>
#include <QMCWaveFunctions/Jastrow/BsplineFunctor.h>
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
enum WaveFunctionTimers
{
  Timer_GL,
  Timer_CompleteUpdates,
};

TimerNameLevelList_t<WaveFunctionTimers> WaveFunctionTimerNames =
    {{Timer_GL, "Kinetic Energy", timer_level_coarse},
     {Timer_CompleteUpdates, "Complete Updates", timer_level_coarse}};


void build_WaveFunction(bool useRef,
                        const SPOSet* spo_main,
                        WaveFunction& WF,
                        ParticleSet& ions,
                        ParticleSet& els,
                        const RandomGenerator<QMCTraits::RealType>& RNG,
                        int delay_rank,
                        bool enableJ3)
{
  using valT = WaveFunction::valT;
  using posT = WaveFunction::posT;

  if (WF.Is_built)
  {
    app_log() << "The wavefunction was built before!" << std::endl;
    return;
  }

  // create a spo view
  auto spo = build_SPOSet_view(useRef, spo_main, 1, 0);

  const int nelup = els.getTotalNum() / 2;

  if (useRef)
  {
    using J1OrbType = miniqmcreference::OneBodyJastrowRef<BsplineFunctor<valT>>;
    using J2OrbType = miniqmcreference::TwoBodyJastrowRef<BsplineFunctor<valT>>;
    using J3OrbType = miniqmcreference::ThreeBodyJastrowRef<PolynomialFunctor3D>;
    using DetType   = miniqmcreference::DiracDeterminantRef<>;

    ions.RSoA = ions.R;
    els.RSoA  = els.R;

    // distance tables
    els.addTable(els, DT_SOA);
    WF.ei_TableID = els.addTable(ions, DT_SOA);

    // determinant component
    WF.nelup  = nelup;
    WF.Det_up = new DetType(spo, 0, delay_rank);
    WF.Det_dn = new DetType(spo, nelup, delay_rank);

    // J1 component
    J1OrbType* J1 = new J1OrbType(ions, els);
    buildJ1(*J1, els.Lattice.WignerSeitzRadius);
    WF.Jastrows.push_back(J1);

    // J2 component
    J2OrbType* J2 = new J2OrbType(els);
    buildJ2(*J2, els.Lattice.WignerSeitzRadius);
    WF.Jastrows.push_back(J2);

    // J3 component
    if (enableJ3)
    {
      J3OrbType* J3 = new J3OrbType(ions, els);
      buildJeeI(*J3, els.Lattice.WignerSeitzRadius);
      WF.Jastrows.push_back(J3);
    }
  }
  else
  {
    using J1OrbType = OneBodyJastrow<BsplineFunctor<valT>>;
    using J2OrbType = TwoBodyJastrow<BsplineFunctor<valT>>;
    using J3OrbType = ThreeBodyJastrow<PolynomialFunctor3D>;
    using DetType   = DiracDeterminant<>;

    ions.RSoA = ions.R;
    els.RSoA  = els.R;

    // distance tables
    els.addTable(els, DT_SOA);
    WF.ei_TableID = els.addTable(ions, DT_SOA);

    // determinant component
    WF.nelup  = nelup;
    WF.Det_up = new DetType(spo, 0, delay_rank);
    WF.Det_dn = new DetType(spo, nelup, delay_rank);

    // J1 component
    J1OrbType* J1 = new J1OrbType(ions, els);
    buildJ1(*J1, els.Lattice.WignerSeitzRadius);
    WF.Jastrows.push_back(J1);

    // J2 component
    J2OrbType* J2 = new J2OrbType(els);
    buildJ2(*J2, els.Lattice.WignerSeitzRadius);
    WF.Jastrows.push_back(J2);

    // J3 component
    if (enableJ3)
    {
      J3OrbType* J3 = new J3OrbType(ions, els);
      buildJeeI(*J3, els.Lattice.WignerSeitzRadius);
      WF.Jastrows.push_back(J3);
    }
  }

  WF.setupTimers();

  WF.Is_built = true;
}

WaveFunction::WaveFunction()
      : FirstTime(true),
        Is_built(false),
        nelup(0),
        ei_TableID(1),
        Det_up(nullptr),
        Det_dn(nullptr),
        LogValue(0.0)
  {}

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
  setup_timers(timers, WaveFunctionTimerNames);
  for (int i = 0; i < Jastrows.size(); i++)
  {
    jastrow_timers.push_back(
        TimerManager.createTimer(Jastrows[i]->WaveFunctionComponentName, timer_level_fine));
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
    {
      jastrow_timers[i]->start();
      LogValue += Jastrows[i]->evaluateLog(P, P.G, P.L);
      jastrow_timers[i]->stop();
    }
    FirstTime = false;
  }
}

WaveFunction::posT WaveFunction::evalGrad(ParticleSet& P, int iat)
{
  posT grad_iat = (iat < nelup ? Det_up->evalGrad(P, iat) : Det_dn->evalGrad(P, iat));

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
  grad       = valT(0);
  valT ratio = (iat < nelup ? Det_up->ratioGrad(P, iat, grad) : Det_dn->ratioGrad(P, iat, grad));

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
  valT ratio = (iat < nelup ? Det_up->ratio(P, iat) : Det_dn->ratio(P, iat));

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
  if (iat < nelup)
    Det_up->acceptMove(P, iat);
  else
    Det_dn->acceptMove(P, iat);

  for (size_t i = 0; i < Jastrows.size(); i++)
  {
    jastrow_timers[i]->start();
    Jastrows[i]->acceptMove(P, iat);
    jastrow_timers[i]->stop();
  }
}

void WaveFunction::completeUpdates()
{
  ScopedTimer local_timer(timers[Timer_CompleteUpdates]);
  Det_up->completeUpdates();
  Det_dn->completeUpdates();
}

void WaveFunction::restore(int iat) {}

void WaveFunction::evaluateGL(ParticleSet& P)
{
  ScopedTimer local_timer(timers[Timer_GL]);

  constexpr valT czero(0);
  P.G = czero;
  P.L = czero;
  Det_up->evaluateGL(P, P.G, P.L);
  Det_dn->evaluateGL(P, P.G, P.L);
  LogValue = Det_up->LogValue + Det_dn->LogValue;

  for (size_t i = 0; i < Jastrows.size(); i++)
  {
    jastrow_timers[i]->start();
    Jastrows[i]->evaluateGL(P, P.G, P.L);
    LogValue += Jastrows[i]->LogValue;
    jastrow_timers[i]->stop();
  }
}

void WaveFunction::evaluateRatios(VirtualParticleSet& VP, std::vector<valT>& ratios)
{
  assert(VP.getTotalNum() == ratios.size());
  if (VP.refPtcl < nelup)
    Det_up->evaluateRatios(VP, ratios);
  else
    Det_dn->evaluateRatios(VP, ratios);

  std::vector<valT> t(ratios.size());
  for (size_t i = 0; i < Jastrows.size(); i++)
  {
    jastrow_timers[i]->start();
    Jastrows[i]->evaluateRatios(VP, t);
    for (int j = 0; j < ratios.size(); ++j)
      ratios[j] *= t[j];
    jastrow_timers[i]->stop();
  }
}

void WaveFunction::flex_evaluateLog(const std::vector<WaveFunction*>& WF_list,
                                    const std::vector<ParticleSet*>& P_list) const
{
  if (!WF_list[0]->FirstTime)
    return;
  else if (P_list.size() > 1)
  {
    constexpr valT czero(0);
    const std::vector<ParticleSet::ParticleGradient_t*> G_list(extract_G_list(P_list));
    const std::vector<ParticleSet::ParticleLaplacian_t*> L_list(extract_L_list(P_list));
    ParticleSet::ParticleValue_t LogValues(P_list.size());

    for (int iw = 0; iw < P_list.size(); iw++)
    {
      *G_list[iw] = czero;
      *L_list[iw] = czero;
    }
    // det up/dn
    std::vector<WaveFunctionComponent*> up_list(extract_up_list(WF_list));
    Det_up->multi_evaluateLog(up_list, P_list, G_list, L_list, LogValues);
    for (int iw = 0; iw < P_list.size(); iw++)
      WF_list[iw]->LogValue = LogValues[iw];
    std::vector<WaveFunctionComponent*> dn_list(extract_dn_list(WF_list));
    Det_dn->multi_evaluateLog(dn_list, P_list, G_list, L_list, LogValues);
    for (int iw = 0; iw < P_list.size(); iw++)
      WF_list[iw]->LogValue += LogValues[iw];
    // Jastrow factors
    for (size_t i = 0; i < Jastrows.size(); i++)
    {
      jastrow_timers[i]->start();
      std::vector<WaveFunctionComponent*> jas_list(extract_jas_list(WF_list, i));
      Jastrows[i]->multi_evaluateLog(jas_list, P_list, G_list, L_list, LogValues);
      for (int iw = 0; iw < P_list.size(); iw++)
        WF_list[iw]->LogValue += LogValues[iw];
      jastrow_timers[i]->stop();
    }
    for (int iw = 0; iw < P_list.size(); iw++)
      WF_list[iw]->FirstTime = false;
  }
  else if(P_list.size()==1)
    WF_list[0]->evaluateLog(*P_list[0]);
}

void WaveFunction::flex_evalGrad(const std::vector<WaveFunction*>& WF_list,
                                 const std::vector<ParticleSet*>& P_list,
                                 int iat,
                                 std::vector<posT>& grad_now) const
{
  if (P_list.size() > 1)
  {
    std::vector<posT> grad_now_det(P_list.size());
    if (iat < nelup)
    {
      std::vector<WaveFunctionComponent*> up_list(extract_up_list(WF_list));
      Det_up->multi_evalGrad(up_list, P_list, iat, grad_now_det);
    }
    else
    {
      std::vector<WaveFunctionComponent*> dn_list(extract_dn_list(WF_list));
      Det_dn->multi_evalGrad(dn_list, P_list, iat, grad_now_det);
    }
    for (int iw = 0; iw < P_list.size(); iw++)
      grad_now[iw] = grad_now_det[iw];

    for (size_t i = 0; i < Jastrows.size(); i++)
    {
      jastrow_timers[i]->start();
      std::vector<posT> grad_now_jas(P_list.size());
      std::vector<WaveFunctionComponent*> jas_list(extract_jas_list(WF_list, i));
      Jastrows[i]->multi_evalGrad(jas_list, P_list, iat, grad_now_jas);
      for (int iw = 0; iw < P_list.size(); iw++)
        grad_now[iw] += grad_now_jas[iw];
      jastrow_timers[i]->stop();
    }
  }
  else if(P_list.size()==1)
    grad_now[0] = WF_list[0]->evalGrad(*P_list[0], iat);
}

void WaveFunction::flex_ratioGrad(const std::vector<WaveFunction*>& WF_list,
                                  const std::vector<ParticleSet*>& P_list,
                                  int iat,
                                  std::vector<valT>& ratios,
                                  std::vector<posT>& grad_new) const
{
  if (P_list.size() > 1)
  {
    std::vector<valT> ratios_det(P_list.size());
    for (int iw = 0; iw < P_list.size(); iw++)
      grad_new[iw] = valT(0);
    if (iat < nelup)
    {
      std::vector<WaveFunctionComponent*> up_list(extract_up_list(WF_list));
      Det_up->multi_ratioGrad(up_list, P_list, iat, ratios_det, grad_new);
    }
    else
    {
      std::vector<WaveFunctionComponent*> dn_list(extract_dn_list(WF_list));
      Det_dn->multi_ratioGrad(dn_list, P_list, iat, ratios_det, grad_new);
    }
    for (int iw = 0; iw < P_list.size(); iw++)
      ratios[iw] = ratios_det[iw];

    for (size_t i = 0; i < Jastrows.size(); i++)
    {
      jastrow_timers[i]->start();
      std::vector<valT> ratios_jas(P_list.size());
      std::vector<WaveFunctionComponent*> jas_list(extract_jas_list(WF_list, i));
      Jastrows[i]->multi_ratioGrad(jas_list, P_list, iat, ratios_jas, grad_new);
      for (int iw = 0; iw < P_list.size(); iw++)
        ratios[iw] *= ratios_jas[iw];
      jastrow_timers[i]->stop();
    }
  }
  else if(P_list.size()==1)
    ratios[0] = WF_list[0]->ratioGrad(*P_list[0], iat, grad_new[0]);
}

void WaveFunction::flex_acceptrestoreMove(const std::vector<WaveFunction*>& WF_list,
                                          const std::vector<ParticleSet*>& P_list,
                                          const std::vector<bool>& isAccepted,
                                          int iat) const
{
  if (P_list.size() > 1)
  {
    if (iat < nelup)
    {
      std::vector<WaveFunctionComponent*> up_list(extract_up_list(WF_list));
      Det_up->multi_acceptrestoreMove(up_list, P_list, isAccepted, iat);
    }
    else
    {
      std::vector<WaveFunctionComponent*> dn_list(extract_dn_list(WF_list));
      Det_dn->multi_acceptrestoreMove(dn_list, P_list, isAccepted, iat);
    }

    for (size_t i = 0; i < Jastrows.size(); i++)
    {
      jastrow_timers[i]->start();
      std::vector<WaveFunctionComponent*> jas_list(extract_jas_list(WF_list, i));
      Jastrows[i]->multi_acceptrestoreMove(jas_list, P_list, isAccepted, iat);
      jastrow_timers[i]->stop();
    }
  }
  else if(P_list.size()==1 && isAccepted[0])
    WF_list[0]->acceptMove(*P_list[0], iat);
}

void WaveFunction::flex_evaluateGL(const std::vector<WaveFunction*>& WF_list,
                                   const std::vector<ParticleSet*>& P_list) const
{
  if (P_list.size() > 1)
  {
    ScopedTimer local_timer(timers[Timer_GL]);

    constexpr valT czero(0);
    const std::vector<ParticleSet::ParticleGradient_t*> G_list(extract_G_list(P_list));
    const std::vector<ParticleSet::ParticleLaplacian_t*> L_list(extract_L_list(P_list));

    for (int iw = 0; iw < P_list.size(); iw++)
    {
      *G_list[iw] = czero;
      *L_list[iw] = czero;
    }
    // det up/dn
    std::vector<WaveFunctionComponent*> up_list(extract_up_list(WF_list));
    Det_up->multi_evaluateGL(up_list, P_list, G_list, L_list);
    for (int iw = 0; iw < P_list.size(); iw++)
      WF_list[iw]->LogValue = up_list[iw]->LogValue;
    std::vector<WaveFunctionComponent*> dn_list(extract_dn_list(WF_list));
    Det_dn->multi_evaluateGL(dn_list, P_list, G_list, L_list);
    for (int iw = 0; iw < P_list.size(); iw++)
      WF_list[iw]->LogValue += dn_list[iw]->LogValue;
    // Jastrow factors
    for (size_t i = 0; i < Jastrows.size(); i++)
    {
      std::vector<WaveFunctionComponent*> jas_list(extract_jas_list(WF_list, i));
      Jastrows[i]->multi_evaluateGL(jas_list, P_list, G_list, L_list);
      for (int iw = 0; iw < P_list.size(); iw++)
        WF_list[iw]->LogValue += jas_list[iw]->LogValue;
    }
  }
  else if(P_list.size()==1)
    WF_list[0]->evaluateGL(*P_list[0]);
}

void WaveFunction::flex_completeUpdates(const std::vector<WaveFunction*>& WF_list) const
{
  if (WF_list.size() > 1)
  {
    ScopedTimer local_timer(timers[Timer_CompleteUpdates]);

    std::vector<WaveFunctionComponent*> up_list(extract_up_list(WF_list));
    Det_up->multi_completeUpdates(up_list);
    std::vector<WaveFunctionComponent*> dn_list(extract_dn_list(WF_list));
    Det_dn->multi_completeUpdates(dn_list);
  }
  else if(WF_list.size()==1)
    WF_list[0]->completeUpdates();
}

const std::vector<WaveFunctionComponent*> WaveFunction::extract_up_list(const std::vector<WaveFunction*>& WF_list) const
{
  std::vector<WaveFunctionComponent*> up_list;
  for (auto it = WF_list.begin(); it != WF_list.end(); it++)
    up_list.push_back((*it)->Det_up);
  return up_list;
}

const std::vector<WaveFunctionComponent*> WaveFunction::extract_dn_list(const std::vector<WaveFunction*>& WF_list) const
{
  std::vector<WaveFunctionComponent*> dn_list;
  for (auto it = WF_list.begin(); it != WF_list.end(); it++)
    dn_list.push_back((*it)->Det_dn);
  return dn_list;
}

const std::vector<WaveFunctionComponent*>
    WaveFunction::extract_jas_list(const std::vector<WaveFunction*>& WF_list, int jas_id) const
{
  std::vector<WaveFunctionComponent*> jas_list;
  for (auto it = WF_list.begin(); it != WF_list.end(); it++)
    jas_list.push_back((*it)->Jastrows[jas_id]);
  return jas_list;
}

} // namespace qmcplusplus
