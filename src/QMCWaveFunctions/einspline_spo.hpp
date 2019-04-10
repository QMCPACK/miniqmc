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
/** @file einspline_spo.hpp
 */
#ifndef QMCPLUSPLUS_EINSPLINE_SPO_HPP
#define QMCPLUSPLUS_EINSPLINE_SPO_HPP
#include <Utilities/Configuration.h>
#include <Utilities/NewTimer.h>
#include <Particle/ParticleSet.h>
#include <Numerics/Spline2/bspline_allocator.hpp>
#include <Numerics/Spline2/MultiBspline.hpp>
#include <Utilities/SIMD/allocator.hpp>
#include "Numerics/OhmmsPETE/OhmmsArray.h"
#include "QMCWaveFunctions/SPOSet.h"
#include <iostream>

namespace qmcplusplus
{
template<typename T>
struct einspline_spo : public SPOSet
{
  struct EvaluateVGHTag
  {};
  struct EvaluateVTag
  {};
  typedef Kokkos::TeamPolicy<Kokkos::Serial, EvaluateVGHTag> policy_vgh_serial_t;
  typedef Kokkos::TeamPolicy<EvaluateVGHTag> policy_vgh_parallel_t;
  typedef Kokkos::TeamPolicy<Kokkos::Serial, EvaluateVTag> policy_v_serial_t;
  typedef Kokkos::TeamPolicy<EvaluateVTag> policy_v_parallel_t;

  typedef typename policy_vgh_serial_t::member_type team_vgh_serial_t;
  typedef typename policy_vgh_parallel_t::member_type team_vgh_parallel_t;
  typedef typename policy_v_serial_t::member_type team_v_serial_t;
  typedef typename policy_v_parallel_t::member_type team_v_parallel_t;


  // Whether to use Serial evaluation or not
  int nSplinesSerialThreshold_V;
  int nSplinesSerialThreshold_VGH;


  /// define the einsplie data object type
  using spline_type     = typename bspline_traits<T, 3>::SplineType;
  using vContainer_type = Kokkos::View<T*>;
  using gContainer_type = Kokkos::View<T * [3], Kokkos::LayoutLeft>;
  using hContainer_type = Kokkos::View<T * [6], Kokkos::LayoutLeft>;
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
  /// if true, is copy.  For reference counting & clean up in Kokkos.
  bool is_copy;

  lattice_type Lattice;
  /// use allocator
  einspline::Allocator myAllocator;
  /// compute engine
  MultiBspline<T> compute_engine;

  Kokkos::View<spline_type*> einsplines;
  Kokkos::View<vContainer_type*> psi;
  Kokkos::View<gContainer_type*> grad;
  Kokkos::View<hContainer_type*> hess;

  //Temporary position for communicating within Kokkos parallel sections.
  PosType tmp_pos;
  /// Timer
  NewTimer* timer;

  /// default constructor
  einspline_spo()
      : nSplinesSerialThreshold_V(512),
        nSplinesSerialThreshold_VGH(128),
        nBlocks(0),
        nSplines(0),
        firstBlock(0),
        lastBlock(0),
        tmp_pos(0),
        Owner(false),
        is_copy(false)
  {
    timer = TimerManager.createTimer("Single-Particle Orbitals", timer_level_fine);
  }
  /// disable copy constructor
  einspline_spo(const einspline_spo& in) = default;
  /// disable copy operator
  einspline_spo& operator=(const einspline_spo& in) = delete;

  /** copy constructor
   * @param in einspline_spo
   * @param team_size number of members in a team
   * @param member_id id of this member in a team
   *
   * Create a view of the big object. A simple blocking & padding  method.
   */
  einspline_spo(const einspline_spo& in, int team_size, int member_id)
      : Owner(false), Lattice(in.Lattice)
  {
    nSplinesSerialThreshold_V   = in.nSplinesSerialThreshold_V;
    nSplinesSerialThreshold_VGH = in.nSplinesSerialThreshold_VGH;
    nSplines                    = in.nSplines;
    nSplinesPerBlock            = in.nSplinesPerBlock;
    nBlocks                     = (in.nBlocks + team_size - 1) / team_size;
    firstBlock                  = nBlocks * member_id;
    lastBlock                   = std::min(in.nBlocks, nBlocks * (member_id + 1));
    nBlocks                     = lastBlock - firstBlock;
    // einsplines.resize(nBlocks);
    einsplines = Kokkos::View<spline_type*>("einsplines", nBlocks);
    for (int i = 0, t = firstBlock; i < nBlocks; ++i, ++t)
      einsplines(i) = in.einsplines(t);
    resize();
    timer = TimerManager.createTimer("Single-Particle Orbitals", timer_level_fine);
  }

  /// destructors
  ~einspline_spo()
  {
    //Note the change in garbage collection here.  The reason for doing this is that by
    //changing einsplines to a view, it's more natural to work by reference than by raw pointer.
    // To maintain current interface, redoing the input types of allocate and destroy to call by references
    //  would need to be propagated all the way down.
    // However, since we've converted the large chunks of memory to views, garbage collection is
    // handled automatically.  Thus, setting the spline_type objects to empty views lets Kokkos handle the Garbage collection.

    if (!is_copy)
    {
      einsplines = Kokkos::View<spline_type*>();
      for (int i = 0; i < psi.extent(0); i++)
      {
        psi(i)  = vContainer_type();
        grad(i) = gContainer_type();
        hess(i) = hContainer_type();
      }
      psi  = Kokkos::View<vContainer_type*>();
      grad = Kokkos::View<gContainer_type*>();
      hess = Kokkos::View<hContainer_type*>();
    }
    //    for (int i = 0; i < nBlocks; ++i)
    //      myAllocator.destroy(einsplines(i));
  }

  /// resize the containers
  void resize()
  {
    //    psi.resize(nBlocks);
    //    grad.resize(nBlocks);
    //    hess.resize(nBlocks);

    psi  = Kokkos::View<vContainer_type*>("Psi", nBlocks);
    grad = Kokkos::View<gContainer_type*>("Grad", nBlocks);
    hess = Kokkos::View<hContainer_type*>("Hess", nBlocks);

    for (int i = 0; i < nBlocks; ++i)
    {
      //psi[i].resize(nSplinesPerBlock);
      //grad[i].resize(nSplinesPerBlock);
      //hess[i].resize(nSplinesPerBlock);

      //Using the "view-of-views" placement-new construct.
      new (&psi(i)) vContainer_type("psi_i", nSplinesPerBlock);
      new (&grad(i)) gContainer_type("grad_i", nSplinesPerBlock);
      new (&hess(i)) hContainer_type("hess_i", nSplinesPerBlock);
    }
  }

  // fix for general num_splines
  void set(int nx, int ny, int nz, int num_splines, int nblocks, bool init_random = true)
  {
    nSplines         = num_splines;
    nBlocks          = nblocks;
    nSplinesPerBlock = num_splines / nblocks;
    firstBlock       = 0;
    lastBlock        = nBlocks;
    if (einsplines.extent(0) == 0)
    {
      Owner = true;
      TinyVector<int, 3> ng(nx, ny, nz);
      PosType start(0);
      PosType end(1);

      //    einsplines.resize(nBlocks);
      einsplines = Kokkos::View<spline_type*>("einsplines", nBlocks);

      RandomGenerator<T> myrandom(11);
      //Array<T, 3> coef_data(nx+3, ny+3, nz+3);
      Kokkos::View<T***> coef_data("coef_data", nx + 3, ny + 3, nz + 3);

      for (int i = 0; i < nBlocks; ++i)
      {
        einsplines(i) =
            *myAllocator.createMultiBspline(T(0), start, end, ng, PERIODIC, nSplinesPerBlock);
        if (init_random)
        {
          for (int j = 0; j < nSplinesPerBlock; ++j)
          {
            // Generate different coefficients for each orbital
            myrandom.generate_uniform(coef_data.data(), coef_data.extent(0)*coef_data.extent(1)*coef_data.extent(2));
            myAllocator.setCoefficientsForOneOrbital(j, coef_data, &einsplines(i));
          }
        }
      }
    }
    resize();
  }

  /** evaluate psi */
  inline void evaluate_v(const PosType& p)
  {
    ScopedTimer local_timer(timer);
    tmp_pos = p;
    compute_engine.copy_A44();
    is_copy = true;
    if (nSplines > nSplinesSerialThreshold_V)
      Kokkos::parallel_for("EinsplineSPO::evalute_v_parallel",
                           policy_v_parallel_t(nBlocks, 1, 32),
                           *this);
    else
      Kokkos::parallel_for("EinsplineSPO::evalute_v_serial", policy_v_serial_t(nBlocks, 1, 32), *this);

    is_copy = false;
    //   auto u = Lattice.toUnit_floor(p);
    //   for (int i = 0; i < nBlocks; ++i)
    //    compute_engine.evaluate_v(&einsplines(i), u[0], u[1], u[2], psi(i).data(), nSplinesPerBlock);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const EvaluateVTag&, const team_v_serial_t& team) const
  {
    int block               = team.league_rank();
    auto u                  = Lattice.toUnit_floor(tmp_pos);
    einsplines(block).coefs = einsplines(block).coefs_view.data();
    compute_engine.evaluate_v(team,
                              &einsplines(block),
                              u[0],
                              u[1],
                              u[2],
                              psi(block).data(),
                              psi(block).extent(0));
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const EvaluateVTag&, const team_v_parallel_t& team) const
  {
    int block               = team.league_rank();
    auto u                  = Lattice.toUnit_floor(tmp_pos);
    einsplines(block).coefs = einsplines(block).coefs_view.data();
    compute_engine.evaluate_v(team,
                              &einsplines(block),
                              u[0],
                              u[1],
                              u[2],
                              psi(block).data(),
                              psi(block).extent(0));
  }


  /** evaluate psi */
  inline void evaluate_v_pfor(const PosType& p)
  {
    auto u = Lattice.toUnit_floor(p);
#pragma omp for nowait
    for (int i = 0; i < nBlocks; ++i)
      compute_engine.evaluate_v(&einsplines(i), u[0], u[1], u[2], psi(i).data(), nSplinesPerBlock);
  }

  /** evaluate psi, grad and lap */
  inline void evaluate_vgl(const PosType& p)
  {
    auto u = Lattice.toUnit_floor(p);
    for (int i = 0; i < nBlocks; ++i)
      compute_engine.evaluate_vgl(&einsplines(i),
                                  u[0],
                                  u[1],
                                  u[2],
                                  psi(i).data(),
                                  grad(i).data(),
                                  hess(i).data(),
                                  nSplinesPerBlock);
  }

  /** evaluate psi, grad and lap */
  inline void evaluate_vgl_pfor(const PosType& p)
  {
    auto u = Lattice.toUnit_floor(p);
#pragma omp for nowait
    for (int i = 0; i < nBlocks; ++i)
      compute_engine.evaluate_vgl(&einsplines(i),
                                  u[0],
                                  u[1],
                                  u[2],
                                  psi(i).data(),
                                  grad(i).data(),
                                  hess(i).data(),
                                  nSplinesPerBlock);
  }

  /** evaluate psi, grad and hess */
  inline void evaluate_vgh(const PosType& p)
  {
    ScopedTimer local_timer(timer);
    tmp_pos = p;

    is_copy = true;
    compute_engine.copy_A44();

    if (nSplines > nSplinesSerialThreshold_VGH)
      Kokkos::parallel_for("EinsplineSPO::evalute_vgh", policy_vgh_parallel_t(nBlocks, 1, 32), *this);
    else
      Kokkos::parallel_for("EinsplineSPO::evalute_vgh", policy_vgh_serial_t(nBlocks, 1, 32), *this);
    is_copy = false;
    //auto u = Lattice.toUnit_floor(p);
    //for (int i = 0; i < nBlocks; ++i)
    //  compute_engine.evaluate_vgh(&einsplines(i), u[0], u[1], u[2],
    //                              psi(i).data(), grad(i).data(), hess(i).data(),
    //                              nSplinesPerBlock);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const EvaluateVGHTag&, const team_vgh_parallel_t& team) const
  {
    int block = team.league_rank();
    auto u    = Lattice.toUnit_floor(tmp_pos);
    compute_engine.evaluate_vgh(team,
                                &einsplines[block],
                                u[0],
                                u[1],
                                u[2],
                                psi(block).data(),
                                grad(block).data(),
                                hess(block).data(),
                                psi(block).extent(0));
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const EvaluateVGHTag&, const team_vgh_serial_t& team) const
  {
    int block = team.league_rank();
    auto u    = Lattice.toUnit_floor(tmp_pos);
    compute_engine.evaluate_vgh(team,
                                &einsplines[block],
                                u[0],
                                u[1],
                                u[2],
                                psi(block).data(),
                                grad(block).data(),
                                hess(block).data(),
                                psi(block).extent(0));
  }
  /** evaluate psi, grad and hess */
  inline void evaluate_vgh_pfor(const PosType& p)
  {
    auto u = Lattice.toUnit_floor(p);
#pragma omp for nowait
    for (int i = 0; i < nBlocks; ++i)
      compute_engine.evaluate_vgh(&einsplines(i),
                                  u[0],
                                  u[1],
                                  u[2],
                                  psi(i).data(),
                                  grad(i).data(),
                                  hess(i).data(),
                                  nSplinesPerBlock);
  }

  void print(std::ostream& os)
  {
    os << "SPO nBlocks=" << nBlocks << " firstBlock=" << firstBlock << " lastBlock=" << lastBlock
       << " nSplines=" << nSplines << " nSplinesPerBlock=" << nSplinesPerBlock << std::endl;
  }
};
} // namespace qmcplusplus

#endif
