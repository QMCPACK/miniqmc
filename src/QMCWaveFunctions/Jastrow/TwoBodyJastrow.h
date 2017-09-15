////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by:
// Jeongnim Kim, jeongnim.kim@intel.com,
//    Intel Corp.
// Amrita Mathuriya, amrita.mathuriya@intel.com,
//    Intel Corp.
// Ye Luo, yeluo@anl.gov,
//    Argonne National Laboratory
//
// File created by:
// Jeongnim Kim, jeongnim.kim@intel.com,
//    Intel Corp.
////////////////////////////////////////////////////////////////////////////////
// -*- C++ -*-
#ifndef QMCPLUSPLUS_TWOBODYJASTROW_OPTIMIZED_SOA_H
#define QMCPLUSPLUS_TWOBODYJASTROW_OPTIMIZED_SOA_H
#include "Configuration.h"
#include "QMCWaveFunctions/WaveFunctionComponentBase.h"
#include "Particle/DistanceTableData.h"
#include <simd/allocator.hpp>
#include <simd/algorithm.hpp>
#include <numeric>

/*!
 * @file TwoBodyJastrow.h
 */

namespace qmcplusplus
{

/** @ingroup WaveFunctionComponent
 *  @brief Specialization for two-body Jastrow function using multiple functors
 *
 * Each pair-type can have distinct function \f$u(r_{ij})\f$.
 * For electrons, distinct pair correlation functions are used
 * for spins up-up/down-down and up-down/down-up.
 *
 * Based on J2OrbitalSoA.h with these considerations
 * - DistanceTableData using SoA containers
 * - support mixed precision: FT::real_type != OHMMS_PRECISION
 * - loops over the groups: elminated PairID
 * - support simd function
 * - double the loop counts
 * - Memory use is O(N).
 */
template <class FT> struct TwoBodyJastrow : public WaveFunctionComponentBase
{
  /// alias FuncType
  using FuncType = FT;
  /// type of each component U, dU, d2U;
  using valT = typename FT::real_type;
  /// element position type
  using posT = TinyVector<valT, OHMMS_DIM>;
  /// use the same container
  using RowContainer = DistanceTableData::RowContainer;

  /// number of particles
  size_t N;
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
  Vector<posT> dUat;
  valT *FirstAddressOfdU, *LastAddressOfdU;
  ///\f$d2Uat[i] = sum_(j) d2u_{i,j}\f$
  Vector<valT> d2Uat;
  valT cur_Uat;
  Kokkos::View<valT*> cur_u, cur_du, cur_d2u;
  Kokkos::View<valT*> old_u, old_du, old_d2u;
  Kokkos::View<valT*> DistCompressed;
  Kokkos::View<int*> DistIndice;
  /// Container for \f$F[ig*NumGroups+jg]\f$
  Kokkos::View<FT*> F;

  int iat, igt;
  const RealType* dist;
  RealType* u;
  RealType* du;
  RealType* d2u;
  int first[2];
  int last[2];

  TwoBodyJastrow(ParticleSet &p);
  ~TwoBodyJastrow();

  /* initialize storage */
  void init(ParticleSet &p);

  /** add functor for (ia,ib) pair */
  void addFunc(int ia, int ib, FT *j);

  RealType evaluateLog(ParticleSet &P, ParticleSet::ParticleGradient_t &G,
                       ParticleSet::ParticleLaplacian_t &L);

  /** recompute internal data assuming distance table is fully ready */
  void recompute(ParticleSet &P);

  ValueType ratio(ParticleSet &P, int iat);
  GradType evalGrad(ParticleSet &P, int iat);
  ValueType ratioGrad(ParticleSet &P, int iat, GradType &grad_iat);
  void acceptMove(ParticleSet &P, int iat);

  /** compute G and L after the sweep
   */
  void evaluateGL(ParticleSet &P, ParticleSet::ParticleGradient_t &G,
                  ParticleSet::ParticleLaplacian_t &L,
                  bool fromscratch = false);

  /*@{ internal compute engines*/
  inline void computeU3(ParticleSet &P, int iat, const RealType *restrict dist,
                        RealType *restrict u, RealType *restrict du,
                        RealType *restrict d2u);
  inline void operator() (const typename Kokkos::TeamPolicy<>::member_type& team) const;

  /** compute gradient
   */
  inline posT accumulateG(const valT *restrict du,
                          const RowContainer &displ) const
  {
    posT grad;
    for (int idim = 0; idim < OHMMS_DIM; ++idim)
    {
      const valT *restrict dX = displ.data(idim);
      valT s                  = valT();

#pragma omp simd reduction(+ : s) aligned(du, dX)
      for (int jat = 0; jat < N; ++jat)
        s += du[jat] * dX[jat];
      grad[idim] = s;
    }
    return grad;
  }

  /** compute gradient and lap
   */
  inline void accumulateGL(const valT *restrict du, const valT *restrict d2u,
                           const RowContainer &displ, posT &grad,
                           valT &lap) const
  {
    constexpr valT lapfac = OHMMS_DIM - RealType(1);
    lap                   = valT(0);
    //#pragma omp simd reduction(+:lap)
    for (int jat = 0; jat < N; ++jat)
      lap += d2u[jat] + lapfac * du[jat];
    for (int idim = 0; idim < OHMMS_DIM; ++idim)
    {
      const valT *restrict dX = displ.data(idim);
      valT s                  = valT();
      //#pragma omp simd reduction(+:s)
      for (int jat = 0; jat < N; ++jat)
        s += du[jat] * dX[jat];
      grad[idim] = s;
    }
  }
};

template <typename FT> TwoBodyJastrow<FT>::TwoBodyJastrow(ParticleSet &p)
{
  init(p);
  FirstTime                 = true;
  KEcorr                    = 0.0;
  WaveFunctionComponentName = "TwoBodyJastrow";
}

template <typename FT> TwoBodyJastrow<FT>::~TwoBodyJastrow() {
  if(F.use_count()==1) {
    for(int i=0; i<NumGroups*NumGroups; i++) {
      F(i) = FT();
    }
  }
}

template <typename FT> void TwoBodyJastrow<FT>::init(ParticleSet &p)
{
  N         = p.getTotalNum();
  NumGroups = p.groups();

  Uat.resize(N);
  dUat.resize(N);
  FirstAddressOfdU = &(dUat[0][0]);
  LastAddressOfdU  = FirstAddressOfdU + dUat.size() * OHMMS_DIM;
  d2Uat.resize(N);
  cur_u = Kokkos::View<valT*>("cur_u",N);
  cur_du = Kokkos::View<valT*>("cur_du",N);
  cur_d2u = Kokkos::View<valT*>("cur_d2u",N);
  old_u = Kokkos::View<valT*>("old_u",N);
  old_du = Kokkos::View<valT*>("old_du",N);
  old_d2u = Kokkos::View<valT*>("old_d2u",N);
  F = Kokkos::View<FT*>("FT",NumGroups * NumGroups);
  for(int i=0; i<NumGroups*NumGroups; i++) {
    new (&F(i)) FT();
  }
  DistCompressed = Kokkos::View<valT*>("DistCompressed",N);
  DistIndice = Kokkos::View<int*>("DistIndice",N);
}

template <typename FT> void TwoBodyJastrow<FT>::addFunc(int ia, int ib, FT *j)
{
  if (ia == ib)
  {
    if (ia == 0) // first time, assign everything
    {
      int ij = 0;
      for (int ig = 0; ig < NumGroups; ++ig)
        for (int jg                   = 0; jg < NumGroups; ++jg, ++ij)
          F[ij] = *j;
    }
  }
  else
  {
    if (N == 2)
    {
      // a very special case, 1 up + 1 down
      // uu/dd was prevented by the builder
      for (int ig = 0; ig < NumGroups; ++ig)
        for (int jg              = 0; jg < NumGroups; ++jg)
          F[ig * NumGroups + jg] = *j;
    }
    else
    {
      // generic case
      F[ia * NumGroups + ib]              = *j;
      if (ia < ib) F[ib * NumGroups + ia] = *j;
    }
  }
  std::stringstream aname;
  aname << ia << ib;
  // ChiesaKEcorrection();
  FirstTime = false;
}

/** intenal function to compute \f$\sum_j u(r_j), du/dr, d2u/dr2\f$
 * @param P particleset
 * @param iat particle index
 * @param dist starting distance
 * @param u starting value
 * @param du starting first deriv
 * @param d2u starting second deriv
 */
template <typename FT>
inline void TwoBodyJastrow<FT>::computeU3(ParticleSet &P, int iat_,
                                          const RealType *restrict dist_,
                                          RealType *restrict u_,
                                          RealType *restrict du_,
                                          RealType *restrict d2u_)
{
  iat = iat_;
  dist = dist_;
  u = u_;
  du = du_;
  d2u = d2u_;

  for (int jg = 0; jg < NumGroups; ++jg)
  {
    first[jg] = P.first(jg);
    last[jg]  = P.last(jg);
  }

  constexpr valT czero(0);
  std::fill_n(u, N, czero);
  std::fill_n(du, N, czero);
  std::fill_n(d2u, N, czero);

  igt = P.GroupID[iat] * NumGroups;
  Kokkos::parallel_for(Kokkos::TeamPolicy<>(NumGroups,1,32), *this);
  //for (int jg = 0; jg < NumGroups; ++jg)
  //{
  //}
}

template<typename FT>
KOKKOS_INLINE_FUNCTION void TwoBodyJastrow<FT>::operator() (const typename Kokkos::TeamPolicy<>::member_type& team) const {
  const int jg = team.league_rank();
  //const FuncType f2(*F[igt + jg]);
  int iStart = first[jg];
  int iEnd   = last[jg];
  F[igt + jg].evaluateVGL(team,iStart, iEnd, dist, u, du, d2u, DistCompressed.data(),
                 DistIndice.data());
}

template <typename FT>
typename TwoBodyJastrow<FT>::ValueType TwoBodyJastrow<FT>::ratio(ParticleSet &P,
                                                                 int iat)
{
  // only ratio, ready to compute it again
  UpdateMode = ORB_PBYP_RATIO;

  const DistanceTableData *d_table = P.DistTables[0];
  const auto dist                  = d_table->Temp_r.data();
  cur_Uat                          = valT(0);
  const int igt                    = P.GroupID[iat] * NumGroups;
  for (int jg = 0; jg < NumGroups; ++jg)
  {
    int iStart = P.first(jg);
    int iEnd   = P.last(jg);
    cur_Uat += F[igt + jg].evaluateV(iStart, iEnd, dist, DistCompressed.data());
  }

  return std::exp(Uat[iat] - cur_Uat);
}

template <typename FT>
typename TwoBodyJastrow<FT>::GradType
TwoBodyJastrow<FT>::evalGrad(ParticleSet &P, int iat)
{
  return GradType(dUat[iat]);
}

template <typename FT>
typename TwoBodyJastrow<FT>::ValueType
TwoBodyJastrow<FT>::ratioGrad(ParticleSet &P, int iat, GradType &grad_iat)
{

  UpdateMode = ORB_PBYP_PARTIAL;

  computeU3(P, iat, P.DistTables[0]->Temp_r.data(), cur_u.data(), cur_du.data(),
            cur_d2u.data());
  Kokkos::fence();
  cur_Uat = simd::accumulate_n(cur_u.data(), N, valT());
  Kokkos::fence();
  DiffVal = Uat[iat] - cur_Uat;
  grad_iat += accumulateG(cur_du.data(), P.DistTables[0]->Temp_dr);
  Kokkos::fence();
  return std::exp(DiffVal);
}

template <typename FT>
void TwoBodyJastrow<FT>::acceptMove(ParticleSet &P, int iat)
{
  // get the old u, du, d2u
  const DistanceTableData *d_table = P.DistTables[0];
  computeU3(P, iat, d_table->Distances[iat], old_u.data(), old_du.data(),
            old_d2u.data());
  Kokkos::fence();
  if (UpdateMode == ORB_PBYP_RATIO)
  { // ratio-only during the move; need to compute derivatives
    const auto dist = d_table->Temp_r.data();
    computeU3(P, iat, dist, cur_u.data(), cur_du.data(), cur_d2u.data());
    Kokkos::fence();
  }

  valT cur_d2Uat(0);
  posT cur_dUat;
  const auto &new_dr = d_table->Temp_dr;
  const auto &old_dr = d_table->Displacements[iat];
  for (int jat = 0; jat < N; jat++)
  {
    constexpr valT lapfac = OHMMS_DIM - RealType(1);
    valT du               = cur_u[jat] - old_u[jat];
    posT newg             = cur_du[jat] * new_dr[jat];
    posT dg               = newg - old_du[jat] * old_dr[jat];
    valT newl             = cur_d2u[jat] + lapfac * cur_du[jat];
    valT dl               = old_d2u[jat] + lapfac * old_du[jat] - newl;
    Uat[jat] += du;
    dUat[jat] -= dg;
    d2Uat[jat] += dl;
    cur_dUat += newg;
    cur_d2Uat -= newl;
  }
  Uat[iat]   = cur_Uat;
  dUat[iat]  = cur_dUat;
  d2Uat[iat] = cur_d2Uat;
}

template <typename FT> void TwoBodyJastrow<FT>::recompute(ParticleSet &P)
{
  const DistanceTableData *d_table = P.DistTables[0];
  for (int ig = 0; ig < NumGroups; ++ig)
  {
    const int igt = ig * NumGroups;
    for (int iat = P.first(ig), last = P.last(ig); iat < last; ++iat)
    {
      printf("Recompute: computeU3 %i\n",iat);
      computeU3(P, iat, d_table->Distances[iat], cur_u.data(), cur_du.data(),
                cur_d2u.data());
      Kokkos::fence();
      Uat[iat] = simd::accumulate_n(cur_u.data(), N, valT());
      posT grad;
      valT lap;
      accumulateGL(cur_du.data(), cur_d2u.data(), d_table->Displacements[iat],
                   grad, lap);
      dUat[iat]  = grad;
      d2Uat[iat] = -lap;
    }
  }
}

template <typename FT>
typename TwoBodyJastrow<FT>::RealType
TwoBodyJastrow<FT>::evaluateLog(ParticleSet &P,
                                ParticleSet::ParticleGradient_t &dG,
                                ParticleSet::ParticleLaplacian_t &dL)
{
  evaluateGL(P, dG, dL, true);
  return LogValue;
}

template <typename FT>
void TwoBodyJastrow<FT>::evaluateGL(ParticleSet &P,
                                    ParticleSet::ParticleGradient_t &G,
                                    ParticleSet::ParticleLaplacian_t &L,
                                    bool fromscratch)
{
  Kokkos::fence();
  printf("A\n");
  if (fromscratch) recompute(P);
  Kokkos::fence();
  printf("B\n");
  LogValue = valT(0);
  for (int iat = 0; iat < N; ++iat)
  {
    LogValue += Uat[iat];
    G[iat] += dUat[iat];
    L[iat] += d2Uat[iat];
  }
  printf("C\n");

  constexpr valT mhalf(-0.5);
  LogValue = mhalf * LogValue;
}
}
#endif
