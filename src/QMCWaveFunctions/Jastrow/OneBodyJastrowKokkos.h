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
#ifndef QMCPLUSPLUS_ONEBODYJASTROW_KOKKOS_H
#define QMCPLUSPLUS_ONEBODYJASTROW_KOKKOS_H
#include "Utilities/Configuration.h"
#include "QMCWaveFunctions/WaveFunctionComponent.h"
#include <Utilities/SIMD/allocator.hpp>
#include <Utilities/SIMD/algorithm.hpp>
#include <numeric>

/*!
 * @file OneBodyJastrow.h
 */

namespace qmcplusplus
{
/** @ingroup WaveFunctionComponent
 *  @brief Specialization for one-body Jastrow function using multiple functors
 */
template<class FT>
struct OneBodyJastrow<Devices::KOKKOS, FT> : public WaveFunctionComponent
{
#ifdef QMC_PARALLEL_JASTROW
  typedef Kokkos::TeamPolicy<> policy_t;
#else
  typedef Kokkos::TeamPolicy<Kokkos::Serial> policy_t;
#endif
  /// alias FuncType
  using FuncType = FT;
  /// type of each component U, dU, d2U;
  using valT = typename FT::real_type;
  /// element position type
  using posT = TinyVector<valT, OHMMS_DIM>;
  /// use the same container
  using RowContainer = DistanceTableData::RowContainer;
  /// table index
  int myTableID;
  /// number of ions
  int Nions;
  /// number of electrons
  int Nelec;
  /// number of groups
  int NumGroups;
  /// reference to the sources (ions)
  const ParticleSet& Ions;

  valT curAt;
  valT curLap;
  posT curGrad;

  ///\f$Vat[i] = sum_(j) u_{i,j}\f$
  Vector<valT> Vat;
  Kokkos::View<valT*> U, dU, d2U;
  Kokkos::View<valT*> DistCompressed;
  Kokkos::View<int*> DistIndice;
  Vector<posT> Grad;
  Vector<valT> Lap;
  /// Container for \f$F[ig*NumGroups+jg]\f$
  typedef Kokkos::Device<Kokkos::DefaultHostExecutionSpace, typename Kokkos::DefaultExecutionSpace::memory_space>
      F_device_type;
  Kokkos::View<FT*, F_device_type> F;

  //Kokkos temporary arrays, a la two body jastrow.
  int iat, igt, jg_hack;
  const RealType* dist;
  int first[2], last[2];
  RealType* u;
  RealType* du;
  RealType* d2u;

  OneBodyJastrow(const ParticleSet& ions, ParticleSet& els) : Ions(ions)
  {
    initalize(els);
    myTableID                 = els.addTable(ions, DT_SOA);
    WaveFunctionComponentName = "OneBodyJastrow";
  }

  OneBodyJastrow(const OneBodyJastrow& rhs) = default;

  ~OneBodyJastrow()
  {
    //   for (int i = 0; i < F.size(); ++i)
    //     if (F[i] != nullptr)
    //       delete F[i];
  }

  /* initialize storage */
  void initalize(ParticleSet& els)
  {
    Nions     = Ions.getTotalNum();
    NumGroups = Ions.getSpeciesSet().getTotalNum();
    int fsize = std::max(NumGroups, 4); //Odd choice.  Why 4?
                                        //Ignore for now and port.

    if (NumGroups > 1 && !Ions.IsGrouped)
    {
      NumGroups = 0;
    }
    Nelec = els.getTotalNum();
    Vat.resize(Nelec);
    Grad.resize(Nelec);
    Lap.resize(Nelec);

    U              = Kokkos::View<valT*>("U", Nions);
    dU             = Kokkos::View<valT*>("dU", Nions);
    d2U            = Kokkos::View<valT*>("d2U", Nions);
    DistCompressed = Kokkos::View<valT*>("DistCompressed", Nions);
    DistIndice     = Kokkos::View<int*>("DistIndice", Nions);

    F = Kokkos::View<FT*, F_device_type>("FT", std::max(NumGroups, 4));
    for (int i = 0; i < fsize; i++)
    {
      new (&F(i)) FT();
    }
  }

  void addFunc(int source_type, FT* afunc, int target_type = -1)
  {
    //if (F[source_type] != nullptr)
    //  delete F[source_type];
    F[source_type] = *afunc;
  }

  void recompute(ParticleSet& P)
  {
    const DistanceTableData& d_ie(*(P.DistTables[myTableID]));
    for (int iat = 0; iat < Nelec; ++iat)
    {
      computeU3(P, iat, d_ie.Distances[iat]);
      Vat[iat] = simd::accumulate_n(U.data(), Nions, valT());
      Lap[iat] = accumulateGL(dU.data(), d2U.data(), d_ie.Displacements[iat], Grad[iat]);
    }
  }

  RealType evaluateLog(ParticleSet& P, ParticleSet::ParticleGradient_t& G, ParticleSet::ParticleLaplacian_t& L)
  {
    evaluateGL(P, G, L, true);
    return LogValue;
  }

  ValueType ratio(ParticleSet& P, int iat)
  {
    UpdateMode = ORB_PBYP_RATIO;
    curAt      = computeU(P.DistTables[myTableID]->Temp_r.data());
    return std::exp(Vat[iat] - curAt);
  }

  inline valT computeU(const valT* dist)
  {
    valT curVat(0);
    if (NumGroups > 0)
    {
      for (int jg = 0; jg < NumGroups; ++jg)
      {
        //  if (F[jg] != nullptr)
        curVat += F[jg].evaluateV(-1, Ions.first(jg), Ions.last(jg), dist, DistCompressed.data());
      }
    }
    else
    {
      for (int c = 0; c < Nions; ++c)
      {
        int gid = Ions.GroupID[c];
        //   if (F[gid] != nullptr)
        curVat += F[gid].evaluate(dist[c]);
      }
    }
    return curVat;
  }

  inline void evaluateGL(ParticleSet& P,
                         ParticleSet::ParticleGradient_t& G,
                         ParticleSet::ParticleLaplacian_t& L,
                         bool fromscratch = false)
  {
    if (fromscratch)
      recompute(P);

    for (size_t iat = 0; iat < Nelec; ++iat)
      G[iat] += Grad[iat];
    for (size_t iat = 0; iat < Nelec; ++iat)
      L[iat] -= Lap[iat];
    LogValue = -simd::accumulate_n(Vat.data(), Nelec, valT());
  }

  /** compute gradient and lap
   * @return lap
   */
  inline valT accumulateGL(const valT* restrict du,
                           const valT* restrict d2u,
                           const RowContainer& displ,
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
  inline void computeU3(ParticleSet& P, int iat_, const valT* dist_)
  {
    if (NumGroups > 0)
    {
      iat  = iat_;
      dist = dist_;
      u    = U.data();
      du   = dU.data();
      d2u  = d2U.data();
      // ions are grouped
      constexpr valT czero(0);
      std::fill_n(U.data(), Nions, czero);
      std::fill_n(dU.data(), Nions, czero);
      std::fill_n(d2U.data(), Nions, czero);

      for (int jg = 0; jg < NumGroups; ++jg)
      {
        /*    F[jg].evaluateVGL(-1,
                           Ions.first(jg),
                           Ions.last(jg),
                           dist,
                           U.data(),
                           dU.data(),
                           d2U.data(),
                           DistCompressed.data(),
                           DistIndice.data());*/
        first[jg] = Ions.first(jg);
        last[jg]  = Ions.last(jg);
        jg_hack   = jg;
        Kokkos::parallel_for(policy_t(1, 1, 32), *this);
      }
    }
    else
    {
      for (int c = 0; c < Nions; ++c)
      {
        int gid = Ions.GroupID[c];
        //   if (F[gid] != nullptr)
        if (true)
        {
          U[c] = F[gid].evaluate(dist[c], dU[c], d2U[c]);
          dU[c] /= dist[c];
        }
      }
    }
  }

  KOKKOS_INLINE_FUNCTION void operator()(const typename policy_t::member_type& team) const
  {
    int jg     = jg_hack;
    int iStart = first[jg];
    int iEnd   = last[jg];
    F[jg].evaluateVGL(team, -1, iStart, iEnd, dist, u, du, d2u, DistCompressed.data(), DistIndice.data());
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

    computeU3(P, iat, P.DistTables[myTableID]->Temp_r.data());
    curLap = accumulateGL(dU.data(), d2U.data(), P.DistTables[myTableID]->Temp_dr, curGrad);
    curAt  = simd::accumulate_n(U.data(), Nions, valT());
    grad_iat += curGrad;
    return std::exp(Vat[iat] - curAt);
  }

  /** Accpted move. Update Vat[iat],Grad[iat] and Lap[iat] */
  void acceptMove(ParticleSet& P, int iat)
  {
    if (UpdateMode == ORB_PBYP_RATIO)
    {
      computeU3(P, iat, P.DistTables[myTableID]->Temp_r.data());
      curLap = accumulateGL(dU.data(), d2U.data(), P.DistTables[myTableID]->Temp_dr, curGrad);
    }

    LogValue += Vat[iat] - curAt;
    Vat[iat]  = curAt;
    Grad[iat] = curGrad;
    Lap[iat]  = curLap;
  }
};

} // namespace qmcplusplus
#endif
