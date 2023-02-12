//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License. See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by:
//
// File created by: Jeongnim Kim, jeongnim.kim@intel.com, Intel Corp.
//////////////////////////////////////////////////////////////////////////////////////
// -*- C++ -*-
#ifndef QMCPLUSPLUS_ONEBODYJASTROW_REF_H
#define QMCPLUSPLUS_ONEBODYJASTROW_REF_H
#include "Utilities/Configuration.h"
#include "QMCWaveFunctions/WaveFunctionComponent.h"
#include <numeric>

/*!
 * @file OneBodyJastrowRef.h
 */

namespace miniqmcreference
{
using namespace qmcplusplus;

/** @ingroup WaveFunctionComponent
 *  @brief Specialization for one-body Jastrow function using multiple functors
 */
template<class FT>
struct OneBodyJastrowRef : public WaveFunctionComponent
{
  /// alias FuncType
  using FuncType = FT;
  /// type of each component U, dU, d2U;
  using valT = typename FT::real_type;
  /// element position type
  using posT = TinyVector<valT, OHMMS_DIM>;
  /// use the same container
  using DisplRow = DistanceTable::DisplRow;
  /// number of ions
  int Nions;
  /// number of electrons
  int Nelec;
  /// number of groups
  int NumGroups;
  /// reference to the sources (ions)
  const ParticleSet& Ions;
  /// table index
  const int myTableID;

  valT curAt;
  valT curLap;
  posT curGrad;

  ///\f$Vat[i] = sum_(j) u_{i,j}\f$
  Vector<valT> Vat;
  aligned_vector<valT> U, dU, d2U;
  aligned_vector<valT> DistCompressed;
  aligned_vector<int> DistIndice;
  Vector<posT> Grad;
  Vector<valT> Lap;
  /// Container for \f$F[ig*NumGroups+jg]\f$
  std::vector<FT*> F;

  OneBodyJastrowRef(const ParticleSet& ions, ParticleSet& els) : Ions(ions), myTableID(els.addTable(ions))
  {
    initalize(els);
    WaveFunctionComponentName = "OneBodyJastrowRef";
  }

  OneBodyJastrowRef(const OneBodyJastrowRef& rhs) = delete;

  ~OneBodyJastrowRef()
  {
    for (int i = 0; i < F.size(); ++i)
      if (F[i] != nullptr)
        delete F[i];
  }

  /* initialize storage */
  void initalize(ParticleSet& els)
  {
    Nions     = Ions.getTotalNum();
    NumGroups = Ions.getSpeciesSet().getTotalNum();
    F.resize(std::max(NumGroups, 4), nullptr);

    Nelec = els.getTotalNum();
    Vat.resize(Nelec);
    Grad.resize(Nelec);
    Lap.resize(Nelec);

    U.resize(Nions);
    dU.resize(Nions);
    d2U.resize(Nions);
    DistCompressed.resize(Nions);
    DistIndice.resize(Nions);
  }

  void addFunc(int source_type, FT* afunc, int target_type = -1)
  {
    if (F[source_type] != nullptr)
      delete F[source_type];
    F[source_type] = afunc;
  }

  void recompute(ParticleSet& P)
  {
    const auto& d_ie = P.getDistTableAB(myTableID);
    for (int iat = 0; iat < Nelec; ++iat)
    {
      computeU3(P, iat, d_ie.getDistRow(iat).data());
      Vat[iat] = std::accumulate(U.begin(), U.begin() + Nions, valT());
      Lap[iat] = accumulateGL(dU.data(), d2U.data(), d_ie.getDisplRow(iat), Grad[iat]);
    }
  }

  RealType evaluateLog(ParticleSet& P,
                       ParticleSet::ParticleGradient& G,
                       ParticleSet::ParticleLaplacian& L)
  {
    evaluateGL(P, G, L, true);
    return LogValue;
  }

  ValueType ratio(ParticleSet& P, int iat)
  {
    UpdateMode = ORB_PBYP_RATIO;
    curAt      = computeU(P.getDistTableAB(myTableID).getTempDists().data());
    return std::exp(Vat[iat] - curAt);
  }

  inline void evaluateRatios(VirtualParticleSet& VP, std::vector<ValueType>& ratios)
  {
    for (int k = 0; k < ratios.size(); ++k)
      ratios[k] = std::exp(Vat[VP.refPtcl] - computeU(VP.getDistTableAB(myTableID).getDistRow(k).data()));
  }

  inline valT computeU(const valT* dist)
  {
    valT curVat(0);
    if (NumGroups > 0)
    {
      for (int jg = 0; jg < NumGroups; ++jg)
      {
        if (F[jg] != nullptr)
          curVat += F[jg]->evaluateV(-1, Ions.first(jg), Ions.last(jg), dist, DistCompressed.data());
      }
    }
    else
    {
      for (int c = 0; c < Nions; ++c)
      {
        int gid = Ions.GroupID[c];
        if (F[gid] != nullptr)
          curVat += F[gid]->evaluate(dist[c]);
      }
    }
    return curVat;
  }

  inline void evaluateGL(ParticleSet& P,
                         ParticleSet::ParticleGradient& G,
                         ParticleSet::ParticleLaplacian& L,
                         bool fromscratch = false)
  {
    if (fromscratch)
      recompute(P);

    for (size_t iat = 0; iat < Nelec; ++iat)
      G[iat] += Grad[iat];
    for (size_t iat = 0; iat < Nelec; ++iat)
      L[iat] -= Lap[iat];
    LogValue = -std::accumulate(Vat.begin(), Vat.begin() + Nelec, valT());
  }

  /** compute gradient and lap
   * @return lap
   */
  inline valT accumulateGL(const valT* restrict du,
                           const valT* restrict d2u,
                           const DisplRow& displ,
                           posT& grad) const
  {
    valT lap(0);
    constexpr valT lapfac = OHMMS_DIM - RealType(1);
    for (int jat = 0; jat < Nions; ++jat)
      lap += d2u[jat] + lapfac * du[jat];
    for (int idim = 0; idim < OHMMS_DIM; ++idim)
    {
      const valT* restrict dX = displ.data(idim);
      valT s                  = valT();
      for (int jat = 0; jat < Nions; ++jat)
        s += du[jat] * dX[jat];
      grad[idim] = s;
    }
    return lap;
  }

  /** compute U, dU and d2U
   * @param P quantum particleset
   * @param iat the moving particle
   * @param dist starting address of the distances of the ions wrt the iat-th
   * particle
   */
  inline void computeU3(ParticleSet& P, int iat, const valT* dist)
  {
    if (NumGroups > 0)
    { // ions are grouped
      constexpr valT czero(0);
      std::fill_n(U.data(), Nions, czero);
      std::fill_n(dU.data(), Nions, czero);
      std::fill_n(d2U.data(), Nions, czero);

      for (int jg = 0; jg < NumGroups; ++jg)
      {
        if (F[jg] == nullptr)
          continue;
        F[jg]->evaluateVGL(-1,
                           Ions.first(jg),
                           Ions.last(jg),
                           dist,
                           U.data(),
                           dU.data(),
                           d2U.data(),
                           DistCompressed.data(),
                           DistIndice.data());
      }
    }
    else
    {
      for (int c = 0; c < Nions; ++c)
      {
        int gid = Ions.GroupID[c];
        if (F[gid] != nullptr)
        {
          U[c] = F[gid]->evaluate(dist[c], dU[c], d2U[c]);
          dU[c] /= dist[c];
        }
      }
    }
  }

  /** compute the gradient during particle-by-particle update
   * @param P quantum particleset
   * @param iat particle index
   */
  GradType evalGrad(ParticleSet& P, int iat) { return GradType(Grad[iat]); }

  /** compute the gradient during particle-by-particle update
   * @param P quantum particleset
   * @param iat particle index
   *
   * Using Temp_r. curAt, curGrad and curLap are computed.
   */
  ValueType ratioGrad(ParticleSet& P, int iat, GradType& grad_iat)
  {
    UpdateMode = ORB_PBYP_PARTIAL;

    computeU3(P, iat, P.getDistTableAB(myTableID).getTempDists().data());
    curLap = accumulateGL(dU.data(), d2U.data(), P.getDistTableAB(myTableID).getTempDispls(), curGrad);
    curAt  = std::accumulate(U.begin(), U.begin() + Nions, valT());
    grad_iat += curGrad;
    return std::exp(Vat[iat] - curAt);
  }

  /** Accpted move. Update Vat[iat],Grad[iat] and Lap[iat] */
  void acceptMove(ParticleSet& P, int iat)
  {
    if (UpdateMode == ORB_PBYP_RATIO)
    {
      computeU3(P, iat, P.getDistTableAB(myTableID).getTempDists().data());
      curLap = accumulateGL(dU.data(), d2U.data(), P.getDistTableAB(myTableID).getTempDispls(), curGrad);
    }

    LogValue += Vat[iat] - curAt;
    Vat[iat]  = curAt;
    Grad[iat] = curGrad;
    Lap[iat]  = curLap;
  }
};

} // namespace miniqmcreference
#endif
