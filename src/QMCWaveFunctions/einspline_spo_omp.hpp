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
//
// File created by:
// Jeongnim Kim, jeongnim.kim@intel.com,
//    Intel Corp.
// ////////////////////////////////////////////////////////////////////////////////
// -*- C++ -*-
/** @file einspline_spo_omp.hpp
 */
#ifndef QMCPLUSPLUS_EINSPLINE_SPO_OMP_HPP
#define QMCPLUSPLUS_EINSPLINE_SPO_OMP_HPP
#include <Utilities/Configuration.h>
#include <Utilities/NewTimer.h>
#include <Particle/ParticleSet.h>
#include <Numerics/Spline2/BsplineAllocator.hpp>
#include <Numerics/Spline2/MultiBspline.hpp>
#include <Numerics/Spline2/MultiBsplineOffload.hpp>
#include <Utilities/SIMD/allocator.hpp>
#include "OpenMP/OMPallocator.hpp"
#include "Numerics/OhmmsPETE/OhmmsArray.h"
#include "QMCWaveFunctions/SPOSet.h"
#include <iostream>

namespace qmcplusplus
{
template<typename T>
struct einspline_spo_omp : public SPOSet
{
  /// define the einsplie data object type
  using self_type       = einspline_spo_omp<T>;
  using spline_type     = typename bspline_traits<T, 3>::SplineType;
  using OffloadAlignedAllocator = OMPallocator<T, Mallocator<T, QMC_CLINE>>;
  using OMPMatrix_type = Matrix<T, OffloadAlignedAllocator>;
  using lattice_type    = CrystalLattice<T, 3>;

  /// number of blocks
  int nBlocks;
  /// first logical block index
  int firstBlock;
  /// last gical block index
  int lastBlock;
  /// number of splines
  int nSplines;
  /// number of splines per block
  int nSplinesPerBlock;
  /// if true, responsible for cleaning up einsplines
  bool Owner;
  lattice_type Lattice;
  /// use allocator
  BsplineAllocator<T> myAllocator;

  Vector<spline_type*, OMPallocator<spline_type*>> einsplines;
  ///thread private ratios for reduction when using nested threading, numVP x numThread
  std::vector<OMPMatrix_type> ratios_private;
  ///offload scratch space, dynamically resized to the maximal need
  std::vector<OMPMatrix_type> offload_scratch;

  // for shadows
  Vector<T*, OMPallocator<T*>> offload_scratch_shadows;
  Vector<T, OMPallocator<T>> pos_scratch;

  /// Timer
  NewTimer* timer;

  /// default constructor
  einspline_spo_omp() : nBlocks(0), nSplines(0), firstBlock(0), lastBlock(0), Owner(false)
  {
    timer = TimerManager.createTimer("Single-Particle Orbitals", timer_level_fine);
  }
  /// disable copy constructor
  einspline_spo_omp(const einspline_spo_omp& in) = delete;
  /// disable copy operator
  einspline_spo_omp& operator=(const einspline_spo_omp& in) = delete;

  /** copy constructor
   * @param in einspline_spo_omp
   * @param team_size number of members in a team
   * @param member_id id of this member in a team
   *
   * Create a view of the big object. A simple blocking & padding  method.
   */
  einspline_spo_omp(const einspline_spo_omp& in, int team_size, int member_id) : Owner(false), Lattice(in.Lattice)
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
      auto& tile_ptr               = in.einsplines[t];
      #pragma omp target map(to : i)
      {
        einsplines_ptr[i] = tile_ptr;
      }
#endif
    }
    resize();
    timer = TimerManager.createTimer("Single-Particle Orbitals", timer_level_fine);
  }

  /// destructors
  ~einspline_spo_omp()
  {
    if (Owner)
      for (int i = 0; i < nBlocks; ++i)
        myAllocator.destroy(einsplines[i]);
  }

  /// resize the containers
  void resize()
  {
    ratios_private.resize(nBlocks);
    offload_scratch.resize(nBlocks);
    for (int i = 0; i < nBlocks; ++i)
    {
      offload_scratch[i].resize(10,getAlignedSize<T>(nSplinesPerBlock));
    }
  }

  // fix for general num_splines
  void set(int nx, int ny, int nz, int num_splines, int nblocks, bool init_random = true)
  {
    // setting OrbitalSetSize to num_splines made artificial only in miniQMC
    OrbitalSetSize   = num_splines;

    nSplines         = num_splines;
    nBlocks          = nblocks;
    nSplinesPerBlock = num_splines / nblocks;
    firstBlock       = 0;
    lastBlock        = nBlocks;
    if (einsplines.size()==0)
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
          for (int j = 0; j < nSplinesPerBlock; ++j)
          {
            // Generate different coefficients for each orbital
            myrandom.generate_uniform(coef_data.data(), coef_data.size());
            myAllocator.setCoefficientsForOneOrbital(j, coef_data, einsplines[i]);
          }
        }
#ifdef ENABLE_OFFLOAD
        // attach pointers
        spline_type** restrict einsplines_ptr = einsplines.data();
        spline_type* restrict& tile_ptr       = einsplines[i];
        T* restrict& coefs_ptr                = einsplines[i]->coefs;
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
  inline void evaluate_v(const ParticleSet& P, int iat)
  {
    ScopedTimer local_timer(timer);

    auto u = Lattice.toUnit_floor(P.activeR(iat));

    auto x = u[0];
    auto y = u[1];
    auto z = u[2];
    auto nSplinesPerBlock_local = nSplinesPerBlock;

    const int ChunkSizePerTeam = 128;
    const int NumTeams         = (nSplinesPerBlock + ChunkSizePerTeam - 1) / ChunkSizePerTeam;

    for (int i = 0; i < nBlocks; ++i)
    {
      const auto* restrict spline_m = einsplines[i];
      auto* restrict psi_ptr = offload_scratch[i].data();

#ifdef ENABLE_OFFLOAD
      #pragma omp target teams distribute num_teams(NumTeams) thread_limit(ChunkSizePerTeam) map(always, from:psi_ptr[:nSplinesPerBlock])
#else
      #pragma omp parallel for
#endif
      for (int team_id = 0; team_id < NumTeams; team_id++)
      {
        const int first = ChunkSizePerTeam * team_id;
        const int last  = (first + ChunkSizePerTeam) > nSplinesPerBlock_local ? nSplinesPerBlock_local : first + ChunkSizePerTeam;

        int ix, iy, iz;
        T a[4], b[4], c[4];
        spline2::computeLocationAndFractional(spline_m,
                                              x, y, z,
                                              ix, iy, iz, a, b, c);
#ifdef ENABLE_OFFLOAD
        #pragma omp parallel
#endif
        spline2offload::evaluate_v_v2(spline_m,
                                      ix, iy, iz,
                                      a, b, c,
                                      psi_ptr+first,
                                      first, last);
      }
    }
  }

  inline void evaluate(const ParticleSet& P, int iat, ValueVector_t& psi_v) override
  {
    evaluate_v(P, iat);

    for (int i = 0; i < nBlocks; ++i)
    {
      // in real simulation, phase needs to be applied. Here just fake computation
      const int first = i*nBlocks;
      std::copy_n(offload_scratch[i].data(), std::min((i+1)*nSplinesPerBlock, OrbitalSetSize) - first, psi_v.data()+first);
    }
  }

  void evaluateDetRatios(const VirtualParticleSet& VP,
                         ValueVector_t& psi,
                         const ValueVector_t& psiinv,
                         std::vector<ValueType>& ratios) override
  {
    ScopedTimer local_timer(timer);

    // pack particle positions
    const int nVP = VP.getTotalNum();
    for (int iVP = 0; iVP < nVP; ++iVP)
      ratios[iVP] = ValueType(0);
    pos_scratch.resize(nVP * 3);
    for (int iVP = 0; iVP < nVP; ++iVP)
    {
      auto ru(Lattice.toUnit_floor(VP.activeR(iVP)));
      pos_scratch[iVP * 3]     = ru[0];
      pos_scratch[iVP * 3 + 1] = ru[1];
      pos_scratch[iVP * 3 + 2] = ru[2];
    }
    auto* pos_scratch_ptr = pos_scratch.data();

    auto nBlocks_local = nBlocks;
    auto nSplinesPerBlock_local = nSplinesPerBlock;

    const int ChunkSizePerTeam = 128;
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
      auto* restrict psiinv_ptr          = psiinv.data() + i*nSplinesPerBlock;
      const int actual_block_size = std::min(nSplinesPerBlock, OrbitalSetSize - i*nSplinesPerBlock);

#ifdef ENABLE_OFFLOAD
      #pragma omp target teams distribute collapse(2) num_teams(nVP*NumTeams) thread_limit(ChunkSizePerTeam) \
        map(always, to : pos_scratch_ptr[:pos_scratch.size()], psiinv_ptr[:actual_block_size]) \
        map(always, from: ratios_private_ptr[0:NumTeams*nVP])
#else
      //#pragma omp parallel for
#endif
      for (size_t iVP = 0; iVP < nVP; iVP++)
        for (int team_id = 0; team_id < NumTeams; team_id++)
        {
          const int first = ChunkSizePerTeam * team_id;
          const int last  = (first + ChunkSizePerTeam) > nSplinesPerBlock_local ? nSplinesPerBlock_local : first + ChunkSizePerTeam;
          auto* restrict offload_scratch_iVP_ptr = offload_scratch_ptr + padded_size * iVP;

          int ix, iy, iz;
          T a[4], b[4], c[4], da[4], db[4], dc[4], d2a[4], d2b[4], d2c[4];
          spline2::computeLocationAndFractional(spline_m,
                                                pos_scratch_ptr[iVP*3  ],
                                                pos_scratch_ptr[iVP*3+1],
                                                pos_scratch_ptr[iVP*3+2],
                                                ix, iy, iz,
                                                a, b, c,
                                                da, db, dc,
                                                d2a, d2b, d2c);
          T sum(0);
          PRAGMA_OFFLOAD("omp parallel")
          {
            spline2offload::evaluate_v_v2(spline_m,
                                          ix, iy, iz,
                                          a, b, c,
                                          offload_scratch_iVP_ptr + first,
                                          first, last);
            PRAGMA_OFFLOAD("omp for reduction(+:sum)")
            for (int j = first; j < last; j++)
              sum += offload_scratch_iVP_ptr[j] * psiinv_ptr[j];
          }
          ratios_private_ptr[iVP * NumTeams + team_id] = sum;
        }

      // do the reduction manually
      for (int iVP = 0; iVP < nVP; ++iVP)
        for (int tid = 0; tid < NumTeams; tid++)
          ratios[iVP] += ratios_private[i][iVP][tid];
    }
  }

  /** evaluate psi, grad and lap */
  inline void evaluate_vgl(const ParticleSet& P, int iat)
  {
    auto u = Lattice.toUnit_floor(P.activeR(iat));
    for (int i = 0; i < nBlocks; ++i)
      MultiBsplineEval::evaluate_vgl(einsplines[i], u[0], u[1], u[2], offload_scratch[i][0], offload_scratch[i][1], offload_scratch[i][4],
                                     nSplinesPerBlock);
  }

  /** evaluate psi, grad and hess */
  inline void evaluate_vgh(const ParticleSet& P, int iat)
  {
    ScopedTimer local_timer(timer);

    auto u = Lattice.toUnit_floor(P.activeR(iat));

    auto x = u[0];
    auto y = u[1];
    auto z = u[2];
    auto nSplinesPerBlock_local = nSplinesPerBlock;

    const int ChunkSizePerTeam = 128;
    const int NumTeams         = (nSplinesPerBlock + ChunkSizePerTeam - 1) / ChunkSizePerTeam;

    for (int i = 0; i < nBlocks; ++i)
    {
      const auto* restrict spline_m = einsplines[i];
      auto* restrict offload_scratch_ptr = offload_scratch[i].data();
      int padded_size = getAlignedSize<T>(nSplinesPerBlock);

#ifdef ENABLE_OFFLOAD
      #pragma omp target teams distribute num_teams(NumTeams) thread_limit(ChunkSizePerTeam) \
        map(always, from:offload_scratch_ptr[:10*padded_size])
#else
      #pragma omp parallel for
#endif
      for (int team_id = 0; team_id < NumTeams; team_id++)
      {
        const int first = ChunkSizePerTeam * team_id;
        const int last  = (first + ChunkSizePerTeam) > nSplinesPerBlock_local ? nSplinesPerBlock_local : first + ChunkSizePerTeam;

        int ix, iy, iz;
        T a[4], b[4], c[4], da[4], db[4], dc[4], d2a[4], d2b[4], d2c[4];
        spline2::computeLocationAndFractional(spline_m, x, y, z, ix, iy, iz, a, b, c, da, db, dc, d2a, d2b, d2c);

#ifdef ENABLE_OFFLOAD
        #pragma omp parallel
#endif
        spline2offload::evaluate_vgh_v2(spline_m,
                                        ix, iy, iz,
                                        a, b, c,
                                        da, db, dc,
                                        d2a, d2b, d2c,
                                        offload_scratch_ptr + first,
                                        padded_size,
                                        first, last);
      }
    }
  }

  inline void evaluate(const ParticleSet& P, int iat, ValueVector_t& psi_v, GradVector_t& dpsi_v, ValueVector_t& d2psi_v) override
  {
    evaluate_vgh(P, iat);
    evaluate_build_vgl(psi_v, dpsi_v, d2psi_v);
  }

  inline void evaluate_build_vgl(ValueVector_t& psi_v, GradVector_t& dpsi_v, ValueVector_t& d2psi_v)
  {
    for (int i = 0; i < nBlocks; ++i)
    {
      // in real simulation, phase needs to be applied. Here just fake computation
      const int first = i*nBlocks;
      int padded_size = getAlignedSize<T>(nSplinesPerBlock);

      for (int j = first; j < std::min((i+1)*nSplinesPerBlock, OrbitalSetSize); j++)
      {
        psi_v[j] = offload_scratch[i][0][j-first];
        dpsi_v[j] = GradType(offload_scratch[i][1][j-first], offload_scratch[i][2][j-first], offload_scratch[i][3][j-first]);
        d2psi_v[j] = offload_scratch[i][4][j-first];
      }
    }
  }

  /** evaluate psi, grad and hess of multiple walkers with offload */
  inline void multi_evaluate_vgh(const std::vector<SPOSet*>& spo_list,
                                 const std::vector<ParticleSet*>& P_list, int iat)
  {
    ScopedTimer local_timer(timer);

    const size_t nw = spo_list.size();
    std::vector<self_type*> shadows; shadows.reserve(spo_list.size());
    for (int iw = 0; iw < nw; iw++)
      shadows.push_back(static_cast<self_type*>(spo_list[iw]));

    if (nw * nBlocks != offload_scratch_shadows.size())
    {
      offload_scratch_shadows.resize(nw * nBlocks);
      T** restrict offload_scratch_shadows_ptr  = offload_scratch_shadows.data();

      for (size_t iw = 0; iw < nw; iw++)
      {
        auto& shadow = *shadows[iw];
        for (int i = 0; i < nBlocks; ++i)
        {
          const size_t idx     = iw * nBlocks + i;
          T* restrict offload_scratch_ptr  = shadow.offload_scratch[i].data();
#ifdef ENABLE_OFFLOAD
          //std::cout << "psi_shadows_ptr mapped already? " << omp_target_is_present(psi_shadows_ptr,0) << std::endl;
          //std::cout << "psi_ptr mapped already? " << omp_target_is_present(psi_ptr,0) << std::endl;
          #pragma omp target map(to : idx)
#endif
          {
            offload_scratch_shadows_ptr[idx] = offload_scratch_ptr;
          }
        }
      }
    }
    T** restrict offload_scratch_shadows_ptr = offload_scratch_shadows.data();

    pos_scratch.resize(nw*3);
    for (size_t iw = 0; iw < nw; iw++)
    {
      auto u = Lattice.toUnit_floor(P_list[iw]->activeR(iat));
      pos_scratch[iw*3  ] = u[0];
      pos_scratch[iw*3+1] = u[1];
      pos_scratch[iw*3+2] = u[2];
    }
    //std::cout << "mapped already? " << omp_target_is_present(pos_scratch.data(),0) << std::endl;
    //pos_scratch.update_to_device();

    auto* pos_scratch_ptr = pos_scratch.data();

    auto nBlocks_local = nBlocks;
    auto nSplinesPerBlock_local = nSplinesPerBlock;

    const int ChunkSizePerTeam = 128;
    const int NumTeams         = (nSplinesPerBlock + ChunkSizePerTeam - 1) / ChunkSizePerTeam;

    for (int i = 0; i < nBlocks; ++i)
    {
      const auto* restrict spline_m = einsplines[i];
      int padded_size = getAlignedSize<T>(nSplinesPerBlock);

#ifdef ENABLE_OFFLOAD
      #pragma omp target teams distribute collapse(2) num_teams(nw*NumTeams) thread_limit(ChunkSizePerTeam) \
        map(always, to : pos_scratch_ptr [:pos_scratch.size()])
#else
      #pragma omp parallel for
#endif
      for (size_t iw = 0; iw < nw; iw++)
        for (int team_id = 0; team_id < NumTeams; team_id++)
        {
          const int first = ChunkSizePerTeam * team_id;
          const int last  = (first + ChunkSizePerTeam) > nSplinesPerBlock_local ? nSplinesPerBlock_local : first + ChunkSizePerTeam;

          int ix, iy, iz;
          T a[4], b[4], c[4], da[4], db[4], dc[4], d2a[4], d2b[4], d2c[4];
          spline2::computeLocationAndFractional(spline_m,
                                                pos_scratch_ptr[iw*3  ],
                                                pos_scratch_ptr[iw*3+1],
                                                pos_scratch_ptr[iw*3+2],
                                                ix, iy, iz,
                                                a, b, c,
                                                da, db, dc,
                                                d2a, d2b, d2c);

#ifdef ENABLE_OFFLOAD
        #pragma omp parallel
#endif
          spline2offload::evaluate_vgh_v2(spline_m,
                                          ix, iy, iz,
                                          a, b, c,
                                          da, db, dc,
                                          d2a, d2b, d2c,
                                          offload_scratch_shadows_ptr[iw * nBlocks_local + i] + first,
                                          padded_size,
                                          first, last);
        }

      for (size_t iw = 0; iw < spo_list.size(); iw++)
      {
        auto* offload_scratch_ptr = shadows[iw]->offload_scratch[i].data();
#ifdef ENABLE_OFFLOAD
        #pragma omp target update from(offload_scratch_ptr[:padded_size*10])
#endif
      }
    }
  }

  inline void multi_evaluate(const std::vector<SPOSet*>& spo_list,
                             const std::vector<ParticleSet*>& P_list,
                             int iat,
                             std::vector<ValueVector_t*>& psi_v_list,
                             std::vector<GradVector_t*>& dpsi_v_list,
                             std::vector<ValueVector_t*>& d2psi_v_list) override
  {
    multi_evaluate_vgh(spo_list, P_list, iat);
    for (size_t iw = 0; iw < spo_list.size(); iw++)
      static_cast<self_type*>(spo_list[iw])->evaluate_build_vgl(*psi_v_list[iw], *dpsi_v_list[iw], *d2psi_v_list[iw]);
  }

  void print(std::ostream& os)
  {
    os << "SPO nBlocks=" << nBlocks << " firstBlock=" << firstBlock << " lastBlock=" << lastBlock
       << " nSplines=" << nSplines << " nSplinesPerBlock=" << nSplinesPerBlock << std::endl;
  }
};
} // namespace qmcplusplus

#endif
