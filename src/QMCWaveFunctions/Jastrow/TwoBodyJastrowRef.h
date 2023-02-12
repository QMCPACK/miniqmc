//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License. See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by: Jeongnim Kim, jeongnim.kim@intel.com, Intel Corp.
//                    Amrita Mathuriya, amrita.mathuriya@intel.com, Intel Corp.
//                    Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//
// File created by: Jeongnim Kim, jeongnim.kim@intel.com, Intel Corp.
//////////////////////////////////////////////////////////////////////////////////////
// -*- C++ -*-
#ifndef QMCPLUSPLUS_TWOBODYJASTROW_REF_H
#define QMCPLUSPLUS_TWOBODYJASTROW_REF_H
#include "Utilities/Configuration.h"
#include "QMCWaveFunctions/WaveFunctionComponent.h"
#include "Particle/DistanceTable.h"
#include <numeric>

/*!
 * @file TwoBodyJastrowRef.h
 */

namespace miniqmcreference
{
using namespace qmcplusplus;

/** @ingroup WaveFunctionComponent
 *  @brief Specialization for two-body Jastrow function using multiple functors
 *
 * Each pair-type can have distinct function \f$u(r_{ij})\f$.
 * For electrons, distinct pair correlation functions are used
 * for spins up-up/down-down and up-down/down-up.
 *
 * Based on TwoBodyJastrowRef.h with these considerations
 * - DistanceTable using SoA containers
 * - support mixed precision: FT::real_type != OHMMS_PRECISION
 * - loops over the groups: elminated PairID
 * - support simd function
 * - double the loop counts
 * - Memory use is O(N).
 */
template<class FT>
struct TwoBodyJastrowRef : public WaveFunctionComponent
{
  /// alias FuncType
  using FuncType = FT;
  /// type of each component U, dU, d2U;
  using valT = typename FT::real_type;
  /// element position type
  using posT = TinyVector<valT, OHMMS_DIM>;
  /// use the same container
  using DisplRow = DistanceTable::DisplRow;

  /// number of particles
  size_t N;
  /// number of particles + padded
  size_t N_padded;
  /// number of groups of the target particleset
  size_t NumGroups;
  /// Used to compute correction
  bool FirstTime;
  /// diff value
  RealType DiffVal;
  /// Correction
  RealType KEcorr;
  ///\f$Uat[i] = sum_(j) u_{i,j}\f$
  Vector<valT> Uat;
  ///\f$dUat[i] = sum_(j) du_{i,j}\f$
  using gContainer_type = VectorSoAContainer<valT, OHMMS_DIM>;
  gContainer_type dUat;
  ///\f$d2Uat[i] = sum_(j) d2u_{i,j}\f$
  Vector<valT> d2Uat;
  valT cur_Uat;
  aligned_vector<valT> cur_u, cur_du, cur_d2u;
  aligned_vector<valT> old_u, old_du, old_d2u;
  aligned_vector<valT> DistCompressed;
  aligned_vector<int> DistIndice;
  /// Container for \f$F[ig*NumGroups+jg]\f$
  std::vector<FT*> F;
  /// Uniquue J2 set for cleanup
  std::map<std::string, FT*> J2Unique;
  /// table index
  const int myTableID;

  TwoBodyJastrowRef(ParticleSet& p);
  TwoBodyJastrowRef(const TwoBodyJastrowRef& rhs) = delete;
  ~TwoBodyJastrowRef();

  /* initialize storage */
  void init(ParticleSet& p);

  /** add functor for (ia,ib) pair */
  void addFunc(int ia, int ib, FT* j);

  RealType evaluateLog(ParticleSet& P,
                       ParticleSet::ParticleGradient& G,
                       ParticleSet::ParticleLaplacian& L);

  /** recompute internal data assuming distance table is fully ready */
  void recompute(ParticleSet& P);

  ValueType ratio(ParticleSet& P, int iat);
  void evaluateRatios(VirtualParticleSet& VP, std::vector<ValueType>& ratios)
  {
    for (int k = 0; k < ratios.size(); ++k)
      ratios[k] = std::exp(Uat[VP.refPtcl] - computeU(VP.getRefPS(), VP.refPtcl, VP.getDistTableAB(myTableID).getDistRow(k).data()));
  }

  GradType evalGrad(ParticleSet& P, int iat);
  ValueType ratioGrad(ParticleSet& P, int iat, GradType& grad_iat);
  void acceptMove(ParticleSet& P, int iat);

  /** compute G and L after the sweep
   */
  void evaluateGL(ParticleSet& P,
                  ParticleSet::ParticleGradient& G,
                  ParticleSet::ParticleLaplacian& L,
                  bool fromscratch = false);

  /*@{ internal compute engines*/
  inline valT computeU(const ParticleSet& P, int iat, const RealType* restrict dist)
  {
    valT curUat(0);
    const int igt = P.GroupID[iat] * NumGroups;
    for (int jg = 0; jg < NumGroups; ++jg)
    {
      const FuncType& f2(*F[igt + jg]);
      int iStart = P.first(jg);
      int iEnd   = P.last(jg);
      curUat += f2.evaluateV(iat, iStart, iEnd, dist, DistCompressed.data());
    }
    return curUat;
  }

  inline void computeU3(const ParticleSet& P,
                        int iat,
                        const RealType* restrict dist,
                        RealType* restrict u,
                        RealType* restrict du,
                        RealType* restrict d2u,
                        bool triangle = false);

  /** compute gradient
   */
  inline posT accumulateG(const valT* restrict du, const DisplRow& displ) const
  {
    posT grad;
    for (int idim = 0; idim < OHMMS_DIM; ++idim)
    {
      const valT* restrict dX = displ.data(idim);
      valT s                  = valT();

      for (int jat = 0; jat < N; ++jat)
        s += du[jat] * dX[jat];
      grad[idim] = s;
    }
    return grad;
  }
  /**@} */
};

template<typename FT>
TwoBodyJastrowRef<FT>::TwoBodyJastrowRef(ParticleSet& p)
  : myTableID(p.addTable(p))
{
  init(p);
  FirstTime                 = true;
  KEcorr                    = 0.0;
  WaveFunctionComponentName = "TwoBodyJastrowRef";
}

template<typename FT>
TwoBodyJastrowRef<FT>::~TwoBodyJastrowRef()
{
  auto it = J2Unique.begin();
  while (it != J2Unique.end())
  {
    delete ((*it).second);
    ++it;
  }
} // need to clean up J2Unique

template<typename FT>
void TwoBodyJastrowRef<FT>::init(ParticleSet& p)
{
  N         = p.getTotalNum();
  N_padded  = getAlignedSize<valT>(N);
  NumGroups = p.groups();

  Uat.resize(N);
  dUat.resize(N);
  d2Uat.resize(N);
  cur_u.resize(N);
  cur_du.resize(N);
  cur_d2u.resize(N);
  old_u.resize(N);
  old_du.resize(N);
  old_d2u.resize(N);
  F.resize(NumGroups * NumGroups, nullptr);
  DistCompressed.resize(N);
  DistIndice.resize(N);
}

template<typename FT>
void TwoBodyJastrowRef<FT>::addFunc(int ia, int ib, FT* j)
{
  if (ia == ib)
  {
    if (ia == 0) // first time, assign everything
    {
      int ij = 0;
      for (int ig = 0; ig < NumGroups; ++ig)
        for (int jg = 0; jg < NumGroups; ++jg, ++ij)
          if (F[ij] == nullptr)
            F[ij] = j;
    }
    else
      F[ia * NumGroups + ib] = j;
  }
  else
  {
    if (N == 2)
    {
      // a very special case, 1 up + 1 down
      // uu/dd was prevented by the builder
      for (int ig = 0; ig < NumGroups; ++ig)
        for (int jg = 0; jg < NumGroups; ++jg)
          F[ig * NumGroups + jg] = j;
    }
    else
    {
      // generic case
      F[ia * NumGroups + ib] = j;
      F[ib * NumGroups + ia] = j;
    }
  }
  std::stringstream aname;
  aname << ia << ib;
  J2Unique[aname.str()] = j;
  FirstTime             = false;
}

/** intenal function to compute \f$\sum_j u(r_j), du/dr, d2u/dr2\f$
 * @param P particleset
 * @param iat particle index
 * @param dist starting distance
 * @param u starting value
 * @param du starting first deriv
 * @param d2u starting second deriv
 */
template<typename FT>
inline void TwoBodyJastrowRef<FT>::computeU3(const ParticleSet& P,
                                             int iat,
                                             const RealType* restrict dist,
                                             RealType* restrict u,
                                             RealType* restrict du,
                                             RealType* restrict d2u,
                                             bool triangle)
{
  const int jelmax = triangle ? iat : N;
  constexpr valT czero(0);
  std::fill_n(u, jelmax, czero);
  std::fill_n(du, jelmax, czero);
  std::fill_n(d2u, jelmax, czero);

  const int igt = P.GroupID[iat] * NumGroups;
  for (int jg = 0; jg < NumGroups; ++jg)
  {
    const FuncType& f2(*F[igt + jg]);
    int iStart = P.first(jg);
    int iEnd   = std::min(jelmax, P.last(jg));
    f2.evaluateVGL(iat, iStart, iEnd, dist, u, du, d2u, DistCompressed.data(), DistIndice.data());
  }
  // u[iat]=czero;
  // du[iat]=czero;
  // d2u[iat]=czero;
}

template<typename FT>
typename TwoBodyJastrowRef<FT>::ValueType TwoBodyJastrowRef<FT>::ratio(ParticleSet& P, int iat)
{
  // only ratio, ready to compute it again
  UpdateMode = ORB_PBYP_RATIO;
  cur_Uat    = computeU(P, iat, P.getDistTableAA(myTableID).getTempDists().data());
  return std::exp(Uat[iat] - cur_Uat);
}

template<typename FT>
typename TwoBodyJastrowRef<FT>::GradType TwoBodyJastrowRef<FT>::evalGrad(ParticleSet& P, int iat)
{
  return GradType(dUat[iat]);
}

template<typename FT>
typename TwoBodyJastrowRef<FT>::ValueType
    TwoBodyJastrowRef<FT>::ratioGrad(ParticleSet& P, int iat, GradType& grad_iat)
{
  UpdateMode = ORB_PBYP_PARTIAL;

  computeU3(P, iat, P.getDistTableAA(myTableID).getTempDists().data(), cur_u.data(), cur_du.data(), cur_d2u.data());
  cur_Uat = std::accumulate(cur_u.begin(), cur_u.begin() + N, valT());
  DiffVal = Uat[iat] - cur_Uat;
  grad_iat += accumulateG(cur_du.data(), P.getDistTableAA(myTableID).getTempDispls());
  return std::exp(DiffVal);
}

template<typename FT>
void TwoBodyJastrowRef<FT>::acceptMove(ParticleSet& P, int iat)
{
  // get the old u, du, d2u
  const auto& d_table = P.getDistTableAA(myTableID);
  computeU3(P, iat, d_table.getDistRow(iat).data(), old_u.data(), old_du.data(), old_d2u.data());
  if (UpdateMode == ORB_PBYP_RATIO)
  { // ratio-only during the move; need to compute derivatives
    computeU3(P, iat, d_table.getTempDists().data(), cur_u.data(), cur_du.data(), cur_d2u.data());
  }

  valT cur_d2Uat(0);
  const auto& new_dr    = d_table.getTempDispls();
  const auto& old_dr    = d_table.getDisplRow(iat);
  constexpr valT lapfac = OHMMS_DIM - RealType(1);
  for (int jat = 0; jat < N; jat++)
  {
    const valT du   = cur_u[jat] - old_u[jat];
    const valT newl = cur_d2u[jat] + lapfac * cur_du[jat];
    const valT dl   = old_d2u[jat] + lapfac * old_du[jat] - newl;
    Uat[jat] += du;
    d2Uat[jat] += dl;
    cur_d2Uat -= newl;
  }
  posT cur_dUat;
  for (int idim = 0; idim < OHMMS_DIM; ++idim)
  {
    const valT* restrict new_dX    = new_dr.data(idim);
    const valT* restrict old_dX    = old_dr.data(idim);
    const valT* restrict cur_du_pt = cur_du.data();
    const valT* restrict old_du_pt = old_du.data();
    valT* restrict save_g          = dUat.data(idim);
    valT cur_g                     = cur_dUat[idim];
    for (int jat = 0; jat < N; jat++)
    {
      const valT newg = cur_du_pt[jat] * new_dX[jat];
      const valT dg   = newg - old_du_pt[jat] * old_dX[jat];
      save_g[jat] -= dg;
      cur_g += newg;
    }
    cur_dUat[idim] = cur_g;
  }
  LogValue += Uat[iat] - cur_Uat;
  Uat[iat]   = cur_Uat;
  dUat(iat)  = cur_dUat;
  d2Uat[iat] = cur_d2Uat;
}

template<typename FT>
void TwoBodyJastrowRef<FT>::recompute(ParticleSet& P)
{
  const auto& d_table = P.getDistTableAA(myTableID);
  for (int ig = 0; ig < NumGroups; ++ig)
  {
    const int igt = ig * NumGroups;
    for (int iat = P.first(ig), last = P.last(ig); iat < last; ++iat)
    {
      computeU3(P, iat, d_table.getDistRow(iat).data(), cur_u.data(), cur_du.data(), cur_d2u.data(), true);
      Uat[iat] = std::accumulate(cur_u.begin(), cur_u.begin() + iat, valT());
      posT grad;
      valT lap(0);
      const valT* restrict u    = cur_u.data();
      const valT* restrict du   = cur_du.data();
      const valT* restrict d2u  = cur_d2u.data();
      const DisplRow& displ = d_table.getDisplRow(iat);
      constexpr valT lapfac     = OHMMS_DIM - RealType(1);
      for (int jat = 0; jat < iat; ++jat)
        lap += d2u[jat] + lapfac * du[jat];
      for (int idim = 0; idim < OHMMS_DIM; ++idim)
      {
        const valT* restrict dX = displ.data(idim);
        valT s                  = valT();
        for (int jat = 0; jat < iat; ++jat)
          s += du[jat] * dX[jat];
        grad[idim] = s;
      }
      dUat(iat)  = grad;
      d2Uat[iat] = -lap;
      // add the contribution from the upper triangle
      for (int jat = 0; jat < iat; jat++)
      {
        Uat[jat] += u[jat];
        d2Uat[jat] -= d2u[jat] + lapfac * du[jat];
      }
      for (int idim = 0; idim < OHMMS_DIM; ++idim)
      {
        valT* restrict save_g   = dUat.data(idim);
        const valT* restrict dX = displ.data(idim);
        for (int jat = 0; jat < iat; jat++)
          save_g[jat] -= du[jat] * dX[jat];
      }
    }
  }
}

template<typename FT>
typename TwoBodyJastrowRef<FT>::RealType
    TwoBodyJastrowRef<FT>::evaluateLog(ParticleSet& P,
                                       ParticleSet::ParticleGradient& G,
                                       ParticleSet::ParticleLaplacian& L)
{
  evaluateGL(P, G, L, true);
  return LogValue;
}

template<typename FT>
void TwoBodyJastrowRef<FT>::evaluateGL(ParticleSet& P,
                                       ParticleSet::ParticleGradient& G,
                                       ParticleSet::ParticleLaplacian& L,
                                       bool fromscratch)
{
  if (fromscratch)
    recompute(P);
  LogValue = valT(0);
  for (int iat = 0; iat < N; ++iat)
  {
    LogValue += Uat[iat];
    G[iat] += dUat[iat];
    L[iat] += d2Uat[iat];
  }

  constexpr valT mhalf(-0.5);
  LogValue = mhalf * LogValue;
}

} // namespace miniqmcreference
#endif
