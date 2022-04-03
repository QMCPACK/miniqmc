////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2019 QMCPACK developers.
//
// File developed by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//
// File created by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
// ////////////////////////////////////////////////////////////////////////////////
// -*- C++ -*-
/** @file einspline_spo_omp.cpp
 */

#include "QMCWaveFunctions/einspline_spo_omp.hpp"

#include <iostream>
#include <Numerics/Spline2/MultiBspline.hpp>
#include <Numerics/Spline2/MultiBsplineOffload.hpp>
#include "Numerics/OhmmsPETE/OhmmsArray.h"
#include <Utilities/RandomGenerator.h>

namespace qmcplusplus
{
/// default constructor
template<typename T>
einspline_spo_omp<T>::einspline_spo_omp()
    : nBlocks(0),
      firstBlock(0),
      lastBlock(0),
      nSplines(0),
      Owner(false),
      timer(*timer_manager.createTimer("Single-Particle Orbitals", timer_level_fine))
{}

/// copy constructor
template<typename T>
einspline_spo_omp<T>::einspline_spo_omp(const einspline_spo_omp& in, int team_size, int member_id)
    : Owner(false), Lattice(in.Lattice), timer(*timer_manager.createTimer("Single-Particle Orbitals", timer_level_fine))
{
  OrbitalSetSize   = in.OrbitalSetSize;
  nSplines         = in.nSplines;
  nSplinesPerBlock = in.nSplinesPerBlock;
  nBlocks          = (in.nBlocks + team_size - 1) / team_size;
  firstBlock       = nBlocks * member_id;
  lastBlock        = std::min(in.nBlocks, nBlocks * (member_id + 1));
  nBlocks          = lastBlock - firstBlock;
  einsplines.resize(nBlocks);
  for (int i = 0, t = firstBlock; i < nBlocks; ++i, ++t)
  {
    einsplines[i] = in.einsplines[t];
#ifdef ENABLE_OFFLOAD
    spline_type** einsplines_ptr = einsplines.data();
    spline_type* tile_ptr        = in.einsplines[t];
#pragma omp target map(to : i)
    {
      einsplines_ptr[i] = tile_ptr;
    }
#endif
  }
  resize();
}

/// destructors
template<typename T>
einspline_spo_omp<T>::~einspline_spo_omp()
{
  if (Owner)
    for (int i = 0; i < nBlocks; ++i)
      myAllocator.destroy(einsplines[i]);
}

/// resize the containers
template<typename T>
void einspline_spo_omp<T>::resize()
{
  ratios_private.resize(nBlocks);
  offload_scratch.resize(nBlocks);
  multi_offload_scratch.resize(nBlocks);
  for (int i = 0; i < nBlocks; ++i)
  {
    offload_scratch[i].resize(vgh_dim, getAlignedSize<T>(nSplinesPerBlock));
  }
}

/// If not initialized previously, generate splines coeficients of \p num_splines
/// divided into \p nblocks chunks each with a grid \p nx x \p ny x \p nz.
/// If \p init_random is true, in each chunk, one orbital is fully randomized
/// and others are tweaked based on it.
template<typename T>
void einspline_spo_omp<T>::set(int nx, int ny, int nz, int num_splines, int nblocks, bool init_random)
{
  // setting OrbitalSetSize to num_splines made artificial only in miniQMC
  OrbitalSetSize = num_splines;

  nSplines         = num_splines;
  nBlocks          = nblocks;
  nSplinesPerBlock = num_splines / nblocks;
  firstBlock       = 0;
  lastBlock        = nBlocks;
  if (einsplines.size() == 0)
  {
    Owner = true;
    TinyVector<int, 3> ng(nx, ny, nz);
    PosType start(0);
    PosType end(1);
    einsplines.resize(nBlocks);
    RandomGenerator<T> myrandom(11);
    Array<T, 3> coef_data(nx + 3, ny + 3, nz + 3);
    for (int i = 0; i < nBlocks; ++i)
    {
      einsplines[i] = myAllocator.createMultiBspline(T(0), start, end, ng, PERIODIC, nSplinesPerBlock);
      if (init_random)
      {
        // Generate a orbital fully with fully randomized coefficients
        myrandom.generate_uniform(coef_data.data(), coef_data.size());
        // Generate different coefficients for each orbital by tweaking coef_data
        myAllocator.setCoefficientsForOrbitals(0, nSplinesPerBlock, coef_data, einsplines[i]);
      }
#ifdef ENABLE_OFFLOAD
      // attach pointers
      spline_type** einsplines_ptr = einsplines.data();
      spline_type* tile_ptr        = einsplines[i];
      T* coefs_ptr                 = einsplines[i]->coefs;
#pragma omp target enter data map(to : tile_ptr [0:1])
// Ye: I still don't understand why this line must be separated from the previous one.
#pragma omp target enter data map(to : coefs_ptr [0:einsplines[i]->coefs_size])
//std::cout << "YYYY offload size = " << einsplines[i]->coefs_size << std::endl;
#pragma omp target map(to : i)
      {
        einsplines_ptr[i]        = tile_ptr;
        einsplines_ptr[i]->coefs = coefs_ptr;
      }
      //std::cout << "YYYY end of spline offloading" << std::endl;
#endif
    }
  }
  resize();
}

/** evaluate psi */
template<typename T>
void einspline_spo_omp<T>::evaluate_v(const ParticleSet& P, int iat, bool host_ready)
{
  ScopedTimer local_timer(timer);

  auto u = Lattice.toUnit_floor(P.activeR(iat));

  auto x                      = u[0];
  auto y                      = u[1];
  auto z                      = u[2];
  auto nSplinesPerBlock_local = nSplinesPerBlock;

  const int ChunkSizePerTeam = 192;
  const int NumTeams         = (nSplinesPerBlock + ChunkSizePerTeam - 1) / ChunkSizePerTeam;

  for (int i = 0; i < nBlocks; ++i)
  {
    const auto* restrict spline_m = einsplines[i];
    auto* restrict psi_ptr        = offload_scratch[i].data();

    PRAGMA_OFFLOAD("omp target teams distribute num_teams(NumTeams) thread_limit(ChunkSizePerTeam)")
    for (int team_id = 0; team_id < NumTeams; team_id++)
    {
      const int first = ChunkSizePerTeam * team_id;
      const int last =
          (first + ChunkSizePerTeam) > nSplinesPerBlock_local ? nSplinesPerBlock_local : first + ChunkSizePerTeam;

      int ix, iy, iz;
      T a[4], b[4], c[4];
      spline2::computeLocationAndFractional(spline_m, x, y, z, ix, iy, iz, a, b, c);
      PRAGMA_OFFLOAD("omp parallel for")
      for (int ind = 0; ind < last - first; ind++)
        spline2offload::evaluate_v_v2(spline_m, ix, iy, iz, a, b, c, psi_ptr + first, first, ind);
    }

    if (host_ready)
    {
      PRAGMA_OFFLOAD("omp target update from(psi_ptr[:nSplinesPerBlock])")
    }
  }
}

template<typename T>
inline void einspline_spo_omp<T>::evaluate(const ParticleSet& P, int iat, ValueVector_t& psi_v)
{
  evaluate_v(P, iat);

  for (int i = 0; i < nBlocks; ++i)
  {
    // in real simulation, phase needs to be applied. Here just fake computation
    const int first = i * nBlocks;
    std::copy_n(offload_scratch[i].data(), std::min((i + 1) * nSplinesPerBlock, OrbitalSetSize) - first,
                psi_v.data() + first);
  }
}

template<typename T>
void einspline_spo_omp<T>::evaluateDetRatios(const VirtualParticleSet& VP,
                                             ValueVector_t& psi,
                                             const ValueVector_t& psiinv,
                                             std::vector<ValueType>& ratios)
{
  ScopedTimer local_timer(timer);

  const int nVP = VP.getTotalNum();
  if (psiinv_pos_copy.size() < psiinv.size() + nVP * 3)
    psiinv_pos_copy.resize(psiinv.size() + nVP * 3);
  // stage psiinv to psiinv_pos_copy
  std::copy_n(psiinv.data(), psiinv.size(), psiinv_pos_copy.data());

  // pack particle positions
  auto* restrict pos_scratch = psiinv_pos_copy.data() + psiinv.size();
  for (int iVP = 0; iVP < nVP; ++iVP)
  {
    auto ru(Lattice.toUnit_floor(VP.activeR(iVP)));
    pos_scratch[iVP * 3]     = ru[0];
    pos_scratch[iVP * 3 + 1] = ru[1];
    pos_scratch[iVP * 3 + 2] = ru[2];
  }

  auto nBlocks_local          = nBlocks;
  auto nSplinesPerBlock_local = nSplinesPerBlock;

  const int ChunkSizePerTeam = 192;
  const int NumTeams         = (nSplinesPerBlock + ChunkSizePerTeam - 1) / ChunkSizePerTeam;
  for (int i = 0; i < nBlocks; ++i)
  {
    const auto* restrict spline_m = einsplines[i];
    if (ratios_private[i].size() < NumTeams * nVP)
      ratios_private[i].resize(nVP, NumTeams);
    int padded_size = getAlignedSize<T>(nSplinesPerBlock);
    if (offload_scratch[i].rows() < nVP)
      offload_scratch[i].resize(nVP, padded_size);

    auto* restrict offload_scratch_ptr = offload_scratch[i].data();
    auto* restrict ratios_private_ptr  = ratios_private[i].data();
    const int actual_block_size        = std::min(nSplinesPerBlock, OrbitalSetSize - i * nSplinesPerBlock);
    auto* restrict psiinv_ptr          = psiinv_pos_copy.data();
    const auto psiinv_size             = psiinv.size();

    PRAGMA_OFFLOAD("omp target teams distribute collapse(2) num_teams(nVP* NumTeams) thread_limit(ChunkSizePerTeam) \
                    map(always, to: psiinv_ptr [0:psiinv_pos_copy.size()]) \
                    map(always, from: ratios_private_ptr [0:NumTeams * nVP])")
    for (size_t iVP = 0; iVP < nVP; iVP++)
      for (int team_id = 0; team_id < NumTeams; team_id++)
      {
        const int first = ChunkSizePerTeam * team_id;
        const int last =
            (first + ChunkSizePerTeam) > nSplinesPerBlock_local ? nSplinesPerBlock_local : first + ChunkSizePerTeam;
        auto* restrict offload_scratch_iVP_ptr = offload_scratch_ptr + padded_size * iVP;
        auto* restrict pos_scratch             = psiinv_ptr + psiinv_size;

        int ix, iy, iz;
        T a[4], b[4], c[4];
        spline2::computeLocationAndFractional(spline_m, pos_scratch[iVP * 3], pos_scratch[iVP * 3 + 1],
                                              pos_scratch[iVP * 3 + 2], ix, iy, iz, a, b, c);
        PRAGMA_OFFLOAD("omp parallel for")
        for (int ind = 0; ind < last - first; ind++)
          spline2offload::evaluate_v_v2(spline_m, ix, iy, iz, a, b, c, offload_scratch_iVP_ptr + first, first, ind);
        T sum(0);
        PRAGMA_OFFLOAD("omp parallel for reduction(+:sum)")
        for (int j = first; j < last; j++)
          sum += offload_scratch_iVP_ptr[j] * psiinv_ptr[i * nSplinesPerBlock_local + j];
        ratios_private_ptr[iVP * NumTeams + team_id] = sum;
      }

    // do the reduction manually
    for (int iVP = 0; iVP < nVP; ++iVP)
    {
      ratios[iVP] = ValueType(0);
      for (int tid = 0; tid < NumTeams; tid++)
        ratios[iVP] += ratios_private[i][iVP][tid];
    }
  }
}

/** evaluate psi, grad and lap */
template<typename T>
void einspline_spo_omp<T>::evaluate_vgl(const ParticleSet& P, int iat)
{
  auto u = Lattice.toUnit_floor(P.activeR(iat));
  for (int i = 0; i < nBlocks; ++i)
    MultiBsplineEval::evaluate_vgl(einsplines[i], u[0], u[1], u[2], offload_scratch[i][0], offload_scratch[i][1],
                                   offload_scratch[i][4], nSplinesPerBlock);
}

/** evaluate psi, grad and hess */
template<typename T>
void einspline_spo_omp<T>::evaluate_vgh(const ParticleSet& P, int iat, bool host_ready)
{
  ScopedTimer local_timer(timer);

  auto u = Lattice.toUnit_floor(P.activeR(iat));

  auto x                      = u[0];
  auto y                      = u[1];
  auto z                      = u[2];
  auto nSplinesPerBlock_local = nSplinesPerBlock;

  const int ChunkSizePerTeam = 192;
  const int NumTeams         = (nSplinesPerBlock + ChunkSizePerTeam - 1) / ChunkSizePerTeam;

  for (int i = 0; i < nBlocks; ++i)
  {
    const auto* restrict spline_m      = einsplines[i];
    auto* restrict offload_scratch_ptr = offload_scratch[i].data();
    int padded_size                    = getAlignedSize<T>(nSplinesPerBlock);

    PRAGMA_OFFLOAD("omp target teams distribute num_teams(NumTeams) thread_limit(ChunkSizePerTeam)")
    for (int team_id = 0; team_id < NumTeams; team_id++)
    {
      const int first = ChunkSizePerTeam * team_id;
      const int last =
          (first + ChunkSizePerTeam) > nSplinesPerBlock_local ? nSplinesPerBlock_local : first + ChunkSizePerTeam;

      int ix, iy, iz;
      T a[4], b[4], c[4], da[4], db[4], dc[4], d2a[4], d2b[4], d2c[4];
      spline2::computeLocationAndFractional(spline_m, x, y, z, ix, iy, iz, a, b, c, da, db, dc, d2a, d2b, d2c);

      PRAGMA_OFFLOAD("omp parallel for")
      for (int ind = 0; ind < last - first; ind++)
        spline2offload::evaluate_vgh_v2(spline_m, ix, iy, iz, a, b, c, da, db, dc, d2a, d2b, d2c,
                                        offload_scratch_ptr + first, padded_size, first, ind);
    }

    if (host_ready)
      offload_scratch[i].updateFrom();
  }
}

template<typename T>
inline void einspline_spo_omp<T>::evaluate(const ParticleSet& P,
                                           int iat,
                                           ValueVector_t& psi_v,
                                           GradVector_t& dpsi_v,
                                           ValueVector_t& d2psi_v)
{
  evaluate_vgh(P, iat);
  evaluate_build_vgl(psi_v, dpsi_v, d2psi_v);
}

template<typename T>
void einspline_spo_omp<T>::evaluate_build_vgl(ValueVector_t& psi_v, GradVector_t& dpsi_v, ValueVector_t& d2psi_v)
{
  for (int i = 0; i < nBlocks; ++i)
  {
    // in real simulation, phase needs to be applied. Here just fake computation
    const int first = i * nBlocks;
    int padded_size = getAlignedSize<T>(nSplinesPerBlock);

    for (int j = first; j < std::min((i + 1) * nSplinesPerBlock, OrbitalSetSize); j++)
    {
      psi_v[j]   = offload_scratch[i][0][j - first];
      dpsi_v[j]  = GradType(offload_scratch[i][1][j - first], offload_scratch[i][2][j - first],
                            offload_scratch[i][3][j - first]);
      d2psi_v[j] = offload_scratch[i][4][j - first];
    }
  }
}

/** evaluate psi, grad and hess of multiple walkers with offload */
template<typename T>
void einspline_spo_omp<T>::multi_evaluate_vgh(const std::vector<SPOSet*>& spo_list,
                                              const std::vector<ParticleSet*>& P_list,
                                              int iat,
                                              bool host_ready)
{
  ScopedTimer local_timer(timer);

  const size_t nw = spo_list.size();
  std::vector<self_type*> shadows;
  shadows.reserve(spo_list.size());
  for (int iw = 0; iw < nw; iw++)
    shadows.push_back(static_cast<self_type*>(spo_list[iw]));

  pos_scratch.resize(nw * 3);
  for (size_t iw = 0; iw < nw; iw++)
  {
    auto u                  = Lattice.toUnit_floor(P_list[iw]->activeR(iat));
    pos_scratch[iw * 3]     = u[0];
    pos_scratch[iw * 3 + 1] = u[1];
    pos_scratch[iw * 3 + 2] = u[2];
  }
  //std::cout << "mapped already? " << omp_target_is_present(pos_scratch.data(),0) << std::endl;
  //pos_scratch.update_to_device();

  auto* pos_scratch_ptr = pos_scratch.data();

  auto nBlocks_local          = nBlocks;
  auto nSplinesPerBlock_local = nSplinesPerBlock;

  const int ChunkSizePerTeam = 192;
  const int NumTeams         = (nSplinesPerBlock + ChunkSizePerTeam - 1) / ChunkSizePerTeam;

  for (int i = 0; i < nBlocks; ++i)
  {
    const auto* restrict spline_m = einsplines[i];
    int padded_size               = getAlignedSize<T>(nSplinesPerBlock);

    multi_offload_scratch[i].resize(vgh_dim * nw, padded_size);
    auto* multi_offload_scratch_ptr = multi_offload_scratch[i].data();

    PRAGMA_OFFLOAD("omp target teams distribute collapse(2) num_teams(nw* NumTeams) thread_limit(ChunkSizePerTeam) \
                    map(always, to: pos_scratch_ptr[:pos_scratch.size()])")
    for (size_t iw = 0; iw < nw; iw++)
      for (int team_id = 0; team_id < NumTeams; team_id++)
      {
        const int first = ChunkSizePerTeam * team_id;
        const int last =
            (first + ChunkSizePerTeam) > nSplinesPerBlock_local ? nSplinesPerBlock_local : first + ChunkSizePerTeam;

        int ix, iy, iz;
        T a[4], b[4], c[4], da[4], db[4], dc[4], d2a[4], d2b[4], d2c[4];
        spline2::computeLocationAndFractional(spline_m, pos_scratch_ptr[iw * 3], pos_scratch_ptr[iw * 3 + 1],
                                              pos_scratch_ptr[iw * 3 + 2], ix, iy, iz, a, b, c, da, db, dc, d2a, d2b,
                                              d2c);

        PRAGMA_OFFLOAD("omp parallel for")
        for (int ind = 0; ind < last - first; ind++)
          spline2offload::evaluate_vgh_v2(spline_m, ix, iy, iz, a, b, c, da, db, dc, d2a, d2b, d2c,
                                          multi_offload_scratch_ptr + iw * vgh_dim * padded_size + first, padded_size,
                                          first, ind);
      }

    if (host_ready)
    {
      multi_offload_scratch[i].updateFrom();
      for (size_t iw = 0; iw < nw; iw++)
        std::copy_n(multi_offload_scratch_ptr + iw * vgh_dim * padded_size, padded_size * vgh_dim,
                    shadows[iw]->offload_scratch[i].data());
    }
  }
}

template<typename T>
void einspline_spo_omp<T>::multi_evaluate_ratio_grads(const std::vector<SPOSet*>& spo_list,
                                                      const std::vector<ParticleSet*>& P_list,
                                                      int iat)
{
  ScopedTimer local_timer(timer);

  const size_t nw = spo_list.size();
  std::vector<self_type*> shadows;
  shadows.reserve(spo_list.size());
  for (int iw = 0; iw < nw; iw++)
    shadows.push_back(static_cast<self_type*>(spo_list[iw]));

  pos_scratch.resize(nw * 3);
  for (size_t iw = 0; iw < nw; iw++)
  {
    auto u                  = Lattice.toUnit_floor(P_list[iw]->activeR(iat));
    pos_scratch[iw * 3]     = u[0];
    pos_scratch[iw * 3 + 1] = u[1];
    pos_scratch[iw * 3 + 2] = u[2];
  }
  //std::cout << "mapped already? " << omp_target_is_present(pos_scratch.data(),0) << std::endl;
  //pos_scratch.update_to_device();

  auto* pos_scratch_ptr = pos_scratch.data();

  auto nBlocks_local          = nBlocks;
  auto nSplinesPerBlock_local = nSplinesPerBlock;

  const int ChunkSizePerTeam = 192;
  const int NumTeams         = (nSplinesPerBlock + ChunkSizePerTeam - 1) / ChunkSizePerTeam;

  for (int i = 0; i < nBlocks; ++i)
  {
    const auto* restrict spline_m = einsplines[i];
    int padded_size               = getAlignedSize<T>(nSplinesPerBlock);

    multi_offload_scratch[i].resize(vgh_dim * nw, padded_size);
    auto* multi_offload_scratch_ptr = multi_offload_scratch[i].data();

    PRAGMA_OFFLOAD("omp target teams distribute collapse(2) num_teams(nw* NumTeams) thread_limit(ChunkSizePerTeam) \
                    map(always, tofrom: pos_scratch_ptr[:pos_scratch.size()])")
    for (size_t iw = 0; iw < nw; iw++)
      for (int team_id = 0; team_id < NumTeams; team_id++)
      {
        const int first = ChunkSizePerTeam * team_id;
        const int last =
            (first + ChunkSizePerTeam) > nSplinesPerBlock_local ? nSplinesPerBlock_local : first + ChunkSizePerTeam;

        int ix, iy, iz;
        T a[4], b[4], c[4], da[4], db[4], dc[4], d2a[4], d2b[4], d2c[4];
        spline2::computeLocationAndFractional(spline_m, pos_scratch_ptr[iw * 3], pos_scratch_ptr[iw * 3 + 1],
                                              pos_scratch_ptr[iw * 3 + 2], ix, iy, iz, a, b, c, da, db, dc, d2a, d2b,
                                              d2c);

        PRAGMA_OFFLOAD("omp parallel for")
        for (int ind = 0; ind < last - first; ind++)
          spline2offload::evaluate_vgh_v2(spline_m, ix, iy, iz, a, b, c, da, db, dc, d2a, d2b, d2c,
                                          multi_offload_scratch_ptr + iw * vgh_dim * padded_size + first, padded_size,
                                          first, ind);

        T ratio(0), grad_x(0), grad_y(0), grad_z(0);
        PRAGMA_OFFLOAD("omp parallel for reduction(+: ratio, grad_x, grad_y, grad_z)")
        for (int ind = first / 2; ind < last / 2; ind++)
        {
          auto* val     = multi_offload_scratch_ptr + iw * vgh_dim * padded_size;
          auto* deriv_x = val + padded_size;
          auto* deriv_y = val + padded_size * 2;
          auto* deriv_z = val + padded_size * 3;
          ratio += val[ind * 2] * val[ind * 2 + 1];
          grad_x += val[ind * 2] * deriv_x[ind * 2] + val[ind * 2 + 1] * deriv_x[ind * 2 + 1];
          grad_y += val[ind * 2] * deriv_y[ind * 2] + val[ind * 2 + 1] * deriv_y[ind * 2 + 1];
          grad_z += val[ind * 2] * deriv_z[ind * 2] + val[ind * 2 + 1] * deriv_z[ind * 2 + 1];
        }
        pos_scratch_ptr[iw * 3]     = grad_x;
        pos_scratch_ptr[iw * 3 + 1] = grad_y;
        pos_scratch_ptr[iw * 3 + 2] = grad_z;
      }
  }
}

template<typename T>
inline void einspline_spo_omp<T>::multi_evaluate(const std::vector<SPOSet*>& spo_list,
                                                 const std::vector<ParticleSet*>& P_list,
                                                 int iat,
                                                 const std::vector<ValueVector_t*>& psi_v_list,
                                                 const std::vector<GradVector_t*>& dpsi_v_list,
                                                 const std::vector<ValueVector_t*>& d2psi_v_list)
{
  multi_evaluate_vgh(spo_list, P_list, iat, true);
  for (size_t iw = 0; iw < spo_list.size(); iw++)
    static_cast<self_type*>(spo_list[iw])->evaluate_build_vgl(*psi_v_list[iw], *dpsi_v_list[iw], *d2psi_v_list[iw]);
}

template<typename T>
void einspline_spo_omp<T>::print(std::ostream& os)
{
  os << "SPO nBlocks=" << nBlocks << " firstBlock=" << firstBlock << " lastBlock=" << lastBlock
     << " nSplines=" << nSplines << " nSplinesPerBlock=" << nSplinesPerBlock << std::endl;
}

template struct einspline_spo_omp<OHMMS_PRECISION>;

} // namespace qmcplusplus
