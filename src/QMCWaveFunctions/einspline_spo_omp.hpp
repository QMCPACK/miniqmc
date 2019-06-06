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
#include "Numerics/OhmmsPETE/OhmmsArray.h"
#include "OpenMP/OMPTinyVector.h"
#include "OpenMP/OMPVector.h"
#include "OpenMP/OMPVectorSoAContainer.h"
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
  using vContainer_type = OMPVector<T, aligned_vector<T>>;
  using gContainer_type = OMPVectorSoAContainer<T, 3>;
  using hContainer_type = OMPVectorSoAContainer<T, 6>;
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

  OMPVector<spline_type*> einsplines;
  std::vector<vContainer_type> psi;
  std::vector<gContainer_type> grad;
  std::vector<hContainer_type> hess;

  // for shadows
  OMPVector<T*> psi_shadows;
  OMPVector<T*> grad_shadows;
  OMPVector<T*> hess_shadows;
  OMPVector<OMPTinyVector<T, 3>> u_shadows;

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
    psi.resize(nBlocks);
    grad.resize(nBlocks);
    hess.resize(nBlocks);
    for (int i = 0; i < nBlocks; ++i)
    {
      psi[i].resize(nSplinesPerBlock);
      grad[i].resize(nSplinesPerBlock);
      hess[i].resize(nSplinesPerBlock);
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
    if (einsplines.empty())
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

    if (nBlocks != psi_shadows.size())
    {
      psi_shadows.resize(nBlocks);

      T** restrict psi_shadows_ptr  = psi_shadows.data();
      for (int i = 0; i < nBlocks; ++i)
      {
        T* restrict psi_ptr  = psi[i].data();
#ifdef ENABLE_OFFLOAD
        #pragma omp target map(to : i)
#endif
        {
          psi_shadows_ptr[i]  = psi_ptr;
        }
      }
    }

    OMPTinyVector<T, 3> u = Lattice.toUnit_floor(P.activeR(iat));

    T** restrict psi_shadows_ptr          = psi_shadows.data();
    spline_type** restrict einsplines_ptr = einsplines.data();

    auto x = u[0];
    auto y = u[1];
    auto z = u[2];
    auto nBlocks_local = nBlocks;
    auto nSplinesPerBlock_local = nSplinesPerBlock;
#ifdef ENABLE_OFFLOAD
    #pragma omp target teams distribute num_teams(nBlocks) thread_limit(nSplinesPerBlock)
#else
    #pragma omp parallel for
#endif
    for (int i = 0; i < nBlocks_local; ++i)
    {
      const auto* restrict spline_m = einsplines_ptr[i];

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
                                    psi_shadows_ptr[i],
                                    nSplinesPerBlock_local);
    }
  }

  inline void evaluate(const ParticleSet& P, int iat, ValueVector_t& psi_v) override
  {
    evaluate_v(P, iat);

    for (int i = 0; i < nBlocks; ++i)
    {
      psi[i].update_from_device();
      // in real simulation, phase needs to be applied. Here just fake computation
      const int first = i*nBlocks;
      std::copy_n(psi[i].data(), std::min((i+1)*nSplinesPerBlock, OrbitalSetSize) - first, psi_v.data()+first);
    }
  }

  /** evaluate psi, grad and lap */
  inline void evaluate_vgl(const ParticleSet& P, int iat)
  {
    auto u = Lattice.toUnit_floor(P.activeR(iat));
    for (int i = 0; i < nBlocks; ++i)
      MultiBsplineEval::evaluate_vgl(einsplines[i], u[0], u[1], u[2], psi[i].data(), grad[i].data(), hess[i].data(),
                                     nSplinesPerBlock);
  }

  /** evaluate psi, grad and hess */
  inline void evaluate_vgh(const ParticleSet& P, int iat)
  {
    ScopedTimer local_timer(timer);

    if (nBlocks != psi_shadows.size())
    {
      psi_shadows.resize(nBlocks);
      grad_shadows.resize(nBlocks);
      hess_shadows.resize(nBlocks);

      T** restrict psi_shadows_ptr  = psi_shadows.data();
      T** restrict grad_shadows_ptr = grad_shadows.data();
      T** restrict hess_shadows_ptr = hess_shadows.data();
      for (int i = 0; i < nBlocks; ++i)
      {
        T* restrict psi_ptr  = psi[i].data();
        T* restrict grad_ptr = grad[i].data();
        T* restrict hess_ptr = hess[i].data();
#ifdef ENABLE_OFFLOAD
        #pragma omp target map(to : i)
#endif
        {
          psi_shadows_ptr[i]  = psi_ptr;
          grad_shadows_ptr[i] = grad_ptr;
          hess_shadows_ptr[i] = hess_ptr;
        }
      }
    }

    OMPTinyVector<T, 3> u = Lattice.toUnit_floor(P.activeR(iat));

    T** restrict psi_shadows_ptr          = psi_shadows.data();
    T** restrict grad_shadows_ptr         = grad_shadows.data();
    T** restrict hess_shadows_ptr         = hess_shadows.data();
    spline_type** restrict einsplines_ptr = einsplines.data();

    auto x = u[0];
    auto y = u[1];
    auto z = u[2];
    auto nBlocks_local = nBlocks;
    auto nSplinesPerBlock_local = nSplinesPerBlock;
#ifdef ENABLE_OFFLOAD
    #pragma omp target teams distribute num_teams(nBlocks) thread_limit(nSplinesPerBlock)
#else
    #pragma omp parallel for
#endif
    for (int i = 0; i < nBlocks_local; ++i)
    {
      const auto* restrict spline_m = einsplines_ptr[i];

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
                                      psi_shadows_ptr[i],
                                      grad_shadows_ptr[i],
                                      hess_shadows_ptr[i],
                                      nSplinesPerBlock_local);
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
      psi[i].update_from_device();
      grad[i].update_from_device();
      hess[i].update_from_device();
      // in real simulation, phase needs to be applied. Here just fake computation
      const int first = i*nBlocks;
      for (int j = first; j < std::min((i+1)*nSplinesPerBlock, OrbitalSetSize); j++)
      {
        psi_v[j] = psi[i][j-first];
        dpsi_v[j] = grad[i][j-first];
        d2psi_v[j] = hess[i].data(0)[j-first] + hess[i].data(1)[j-first] + hess[i].data(2)[j-first] +
                     hess[i].data(3)[j-first] + hess[i].data(4)[j-first] + hess[i].data(5)[j-first];
      }
    }
  }

  void transfer_v_from_device()
  {
    for (int i = 0; i < nBlocks; ++i)
      psi[i].update_from_device();
  }

  void transfer_vgh_from_device()
  {
    for (int i = 0; i < nBlocks; ++i)
    {
      psi[i].update_from_device();
      grad[i].update_from_device();
      hess[i].update_from_device();
    }
  }

  /** evaluate psi, grad and hess of multiple walkers with offload */
  inline void multi_evaluate_vgh(const std::vector<SPOSet*>& spo_list,
                                 const std::vector<ParticleSet*>& P_list, int iat)
  {
    ScopedTimer local_timer(timer);

    const size_t nw = spo_list.size();
    std::vector<self_type*> shadows;
    for (int iw = 0; iw < nw; iw++)
      shadows.push_back(static_cast<self_type*>(spo_list[iw]));

    if (nw * nBlocks != psi_shadows.size())
    {
      psi_shadows.resize(nw * nBlocks);
      grad_shadows.resize(nw * nBlocks);
      hess_shadows.resize(nw * nBlocks);

      T** restrict psi_shadows_ptr  = psi_shadows.data();
      T** restrict grad_shadows_ptr = grad_shadows.data();
      T** restrict hess_shadows_ptr = hess_shadows.data();
      for (size_t iw = 0; iw < nw; iw++)
      {
        auto& shadow = *shadows[iw];
        for (int i = 0; i < nBlocks; ++i)
        {
          const size_t idx     = iw * nBlocks + i;
          T* restrict psi_ptr  = shadow.psi[i].data();
          T* restrict grad_ptr = shadow.grad[i].data();
          T* restrict hess_ptr = shadow.hess[i].data();
#ifdef ENABLE_OFFLOAD
          //std::cout << "psi_shadows_ptr mapped already? " << omp_target_is_present(psi_shadows_ptr,0) << std::endl;
          //std::cout << "psi_ptr mapped already? " << omp_target_is_present(psi_ptr,0) << std::endl;
          #pragma omp target map(to : idx) device(0)
#endif
          {
            psi_shadows_ptr[idx]  = psi_ptr;
            grad_shadows_ptr[idx] = grad_ptr;
            hess_shadows_ptr[idx] = hess_ptr;
          }
        }
      }
    }

    u_shadows.resize(nw);
    for (size_t iw = 0; iw < nw; iw++)
      u_shadows[iw] = Lattice.toUnit_floor(P_list[iw]->activeR(iat));
    //std::cout << "mapped already? " << omp_target_is_present(u_shadows.data(),0) << std::endl;
    //u_shadows.update_to_device();

    T** restrict psi_shadows_ptr          = psi_shadows.data();
    T** restrict grad_shadows_ptr         = grad_shadows.data();
    T** restrict hess_shadows_ptr         = hess_shadows.data();
    spline_type** restrict einsplines_ptr = einsplines.data();
    OMPTinyVector<T, 3>* u_shadows_ptr    = u_shadows.data();

    auto nw_local = nw;
    auto nBlocks_local = nBlocks;
    auto nSplinesPerBlock_local = nSplinesPerBlock;

#ifdef ENABLE_OFFLOAD
    #pragma omp target teams distribute collapse(2) num_teams(nw*nBlocks) thread_limit(nSplinesPerBlock) \
    map(always, to : u_shadows_ptr [0:u_shadows.size()])
#else
    #pragma omp parallel for collapse(2)
#endif
    for (size_t iw = 0; iw < nw_local; iw++)
      for (int i = 0; i < nBlocks_local; ++i)
      {
        const auto* restrict spline_m = einsplines_ptr[i];

        int ix, iy, iz;
        T a[4], b[4], c[4], da[4], db[4], dc[4], d2a[4], d2b[4], d2c[4];
        spline2::computeLocationAndFractional(spline_m,
                                              u_shadows_ptr[iw][0],
                                              u_shadows_ptr[iw][1],
                                              u_shadows_ptr[iw][2],
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
                                        psi_shadows_ptr[iw * nBlocks_local + i],
                                        grad_shadows_ptr[iw * nBlocks_local + i],
                                        hess_shadows_ptr[iw * nBlocks_local + i],
                                        nSplinesPerBlock_local);
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

  void multi_transfer_vgh_from_device(const std::vector<SPOSet*>& spo_list) const
  {
    for (size_t iw = 0; iw < spo_list.size(); iw++)
      (static_cast<self_type*>(spo_list[iw]))->transfer_vgh_from_device();
  }

  void print(std::ostream& os)
  {
    os << "SPO nBlocks=" << nBlocks << " firstBlock=" << firstBlock << " lastBlock=" << lastBlock
       << " nSplines=" << nSplines << " nSplinesPerBlock=" << nSplinesPerBlock << std::endl;
  }
};
} // namespace qmcplusplus

#endif
