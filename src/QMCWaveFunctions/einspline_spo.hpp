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
#include <Particle/ParticleSet.h>
#include <Numerics/Spline2/bspline_allocator.hpp>
#include <Numerics/Spline2/MultiBspline.hpp>
#include <Utilities/SIMD/allocator.hpp>
#include "Numerics/OhmmsPETE/OhmmsArray.h"
#include <iostream>

namespace qmcplusplus
{
template <typename T, typename compute_engine_type = MultiBspline<T> >
struct einspline_spo
{
  struct EvaluateVGHTag {};
  struct EvaluateVTag {};
  typedef Kokkos::TeamPolicy<Kokkos::Serial,EvaluateVGHTag> policy_vgh_serial_t;
  typedef Kokkos::TeamPolicy<EvaluateVGHTag> policy_vgh_parallel_t;
  typedef Kokkos::TeamPolicy<Kokkos::Serial,EvaluateVTag> policy_v_serial_t;
  typedef Kokkos::TeamPolicy<EvaluateVTag> policy_v_parallel_t;

  typedef typename policy_vgh_serial_t::member_type team_vgh_serial_t;
  typedef typename policy_vgh_parallel_t::member_type team_vgh_parallel_t;
  typedef typename policy_v_serial_t::member_type team_v_serial_t;
  typedef typename policy_v_parallel_t::member_type team_v_parallel_t;

  // Whether to use Serial evaluation or not
  int nSplinesSerialThreshold_V;
  int nSplinesSerialThreshold_VGH;

  /// define the einspline data object type
  using spline_type = typename bspline_traits<T, 3>::SplineType;
  using pos_type        = TinyVector<T, 3>;
  using vContainer_type = Kokkos::View<T*>;
  using gContainer_type = Kokkos::View<T*[3], Kokkos::LayoutLeft>;
  using hContainer_type = Kokkos::View<T*[6], Kokkos::LayoutLeft>;
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
  einspline::Allocator myAllocator;
  /// compute engine
  compute_engine_type compute_engine;

  // Needed to be stored in class since we use it as the functor
  pos_type pos;

  // mark instance as copy
  bool is_copy;

  Kokkos::View<spline_type *> einsplines;
  Kokkos::View<vContainer_type*> psi;
  Kokkos::View<gContainer_type*> grad;
  Kokkos::View<hContainer_type*> hess;

  /// default constructor
  einspline_spo()
    : nSplinesSerialThreshold_V(512),nSplinesSerialThreshold_VGH(128),nBlocks(0), nSplines(0), firstBlock(0), lastBlock(0), Owner(false), is_copy(false)
  {
  }
  /// disable copy constructor
  //einspline_spo(const einspline_spo &in) = delete;
  /// disable copy operator
  //einspline_spo &operator=(const einspline_spo &in) = delete;

  /** copy constructor
   * @param in einspline_spo
   * @param team_size number of members in a team
   * @param member_id id of this member in a team
   *
   * Create a view of the big object. A simple blocking & padding  method.
   */
  einspline_spo(einspline_spo &in, int ncrews, int crewID)
    : Owner(false), Lattice(in.Lattice), is_copy(false)
  {
    nSplinesSerialThreshold_V = in.nSplinesSerialThreshold_V;
    nSplinesSerialThreshold_VGH = in.nSplinesSerialThreshold_VGH;
    nSplines         = in.nSplines;
    nSplinesPerBlock = in.nSplinesPerBlock;
    nBlocks          = (in.nBlocks + ncrews - 1) / ncrews;
    firstBlock       = nBlocks * crewID;
    lastBlock        = std::min(in.nBlocks, nBlocks * (crewID + 1));
    nBlocks          = lastBlock - firstBlock;
    einsplines       = Kokkos::View<spline_type*>("einsplines",nBlocks);
    for (int i = 0, t = firstBlock; i < nBlocks; ++i, ++t)
      einsplines[i] = in.einsplines[t];
    resize();
  }

  /// destructors
  ~einspline_spo()
  {
    if(! is_copy) {
      if (psi.extent(0)) clean();
      einsplines = Kokkos::View<spline_type *>() ;
    }
  }

  void clean()
  {
    for(int i=0; i<psi.extent(0); i++) {
      psi(i) = vContainer_type();
      grad(i) = gContainer_type();
      hess(i) = hContainer_type();
    }
    psi = Kokkos::View<vContainer_type*>()  ;
    grad = Kokkos::View<gContainer_type*>()  ;
    hess = Kokkos::View<hContainer_type*>() ;
  }

  /// resize the containers
  void resize()
  {
    if (nBlocks > psi.size())
    {
      clean();
      //      psi.resize(nBlocks);
      //      grad.resize(nBlocks);
      //      hess.resize(nBlocks);
      //      for (int i = 0; i < nBlocks; ++i)
      //      {
      //        psi[i]  = new vContainer_type(nSplinesPerBlock);
      //        grad[i] = new gContainer_type(nSplinesPerBlock);
      //        hess[i] = new hContainer_type(nSplinesPerBlock);
      //      }

      psi = Kokkos::View<vContainer_type*>("Psi", nBlocks);
      grad = Kokkos::View<gContainer_type*>("Grad", nBlocks);
      hess = Kokkos::View<hContainer_type*>("Hess", nBlocks);
      for (int i = 0; i < psi.extent(0); i++) {
	new (&psi(i)) vContainer_type("Psi_i", nSplinesPerBlock);
	new (&grad(i)) gContainer_type("Grad_i", nSplinesPerBlock);
	new (&hess(i)) hContainer_type("Hess_i", nSplinesPerBlock);
      }
    }
  }

  // fix for general num_splines
  void set(int nx, int ny, int nz, int num_splines, int nblocks,
           bool init_random = true)
  {
    nSplines         = num_splines;
    nBlocks          = nblocks;
    nSplinesPerBlock = num_splines / nblocks;
    firstBlock       = 0;
    lastBlock        = nBlocks;
    if (einsplines.extent(0)==0)
    {
      Owner = true;
      TinyVector<int, 3> ng(nx, ny, nz);
      pos_type start(0);
      pos_type end(1);
      einsplines       = Kokkos::View<spline_type*>("einsplines",nBlocks);
      RandomGenerator<T> myrandom(11);
      Array<T, 3> data(nx, ny, nz);
      std::fill(data.begin(), data.end(), T());
      myrandom.generate_uniform(data.data(), data.size());
      for (int i = 0; i < nBlocks; ++i)
      {
        einsplines[i] = *myAllocator.createMultiBspline(T(0), start, end, ng, PERIODIC, nSplinesPerBlock);
        if (init_random)
          for (int j = 0; j < nSplinesPerBlock; ++j)
            myAllocator.set(data.data(), &einsplines[i], j);
      }
    }
    resize();
  }

  /** evaluate psi */
  inline void evaluate_v(const pos_type &p)
  {
    pos = p;
    is_copy = true;
    compute_engine.copy_A44();
    if(nSplines > nSplinesSerialThreshold_V)
      Kokkos::parallel_for("EinsplineSPO::evalute_v_parallel",policy_v_parallel_t(nBlocks,1,32),*this);
    else
      Kokkos::parallel_for("EinsplineSPO::evalute_v_serial",policy_v_serial_t(nBlocks,1,32),*this);
    is_copy = false;
  }
  KOKKOS_INLINE_FUNCTION
  void operator() (const EvaluateVTag&, const team_v_serial_t& team ) const {
    int block = team.league_rank();
    // Need KokkosInlineFunction on Tensor and TinyVector ....
    auto u = Lattice.toUnit(pos);

    compute_engine.evaluate_v(team,&einsplines[block], u[0], u[1], u[2],
                              psi(block).data(), psi(block).extent(0));
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const EvaluateVTag&, const team_v_parallel_t& team ) const {
    int block = team.league_rank();
    // Need KokkosInlineFunction on Tensor and TinyVector ....
    auto u = Lattice.toUnit(pos);

    compute_engine.evaluate_v(team,&einsplines[block], u[0], u[1], u[2],
                              psi(block).data(), psi(block).extent(0));
  }

  /** evaluate psi */
  inline void evaluate_v_pfor(const pos_type &p)
  {
    auto u = Lattice.toUnit_floor(p);
    #pragma omp for nowait
    for (int i = 0; i < nBlocks; ++i)
      compute_engine.evaluate_v(&einsplines[i], u[0], u[1], u[2], &psi(i,0), psi.extent(0));
  }

  /** evaluate psi, grad and lap */
  inline void evaluate_vgl(const pos_type &p)
  {
    auto u = Lattice.toUnit_floor(p);
    for (int i = 0; i < nBlocks; ++i)
      compute_engine.evaluate_vgl(&einsplines[i], u[0], u[1], u[2],
                                  psi(i)->data(), grad(i)->data(), hess(i)->data(),
                                  psi(i)->extent(0));
  }

  /** evaluate psi, grad and lap */
  inline void evaluate_vgl_pfor(const pos_type &p)
  {
    auto u = Lattice.toUnit_floor(p);
    #pragma omp for nowait
    for (int i = 0; i < nBlocks; ++i)
      compute_engine.evaluate_vgl(&einsplines[i], u[0], u[1], u[2],
                                  psi(i)->data(), grad(i)->data(), hess(i)->data(),
                                  psi(i)->extent(0));
  }

  /** evaluate psi, grad and hess */
  inline void evaluate_vgh(const pos_type &p)
  {
    pos = p;
    is_copy = true;
    compute_engine.copy_A44();

    if(nSplines > nSplinesSerialThreshold_VGH)
      Kokkos::parallel_for("EinsplineSPO::evalute_vgh",policy_vgh_parallel_t(nBlocks,1,32),*this);
    else
      Kokkos::parallel_for("EinsplineSPO::evalute_vgh",policy_vgh_serial_t(nBlocks,1,32),*this);
    is_copy = false;
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const EvaluateVGHTag&, const team_vgh_parallel_t& team ) const {
    int block = team.league_rank();
    auto u = Lattice.toUnit(pos);
    compute_engine.evaluate_vgh(team,&einsplines[block], u[0], u[1], u[2],
                                      psi(block).data(), grad(block).data(), hess(block).data(),
                                      psi(block).extent(0));
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const EvaluateVGHTag&, const team_vgh_serial_t& team ) const {
    int block = team.league_rank();
    auto u = Lattice.toUnit(pos);
    compute_engine.evaluate_vgh(team,&einsplines[block], u[0], u[1], u[2],
                                      psi(block).data(), grad(block).data(), hess(block).data(),
                                      psi(block).extent(0));
  }

  /** evaluate psi, grad and hess */
  inline void evaluate_vgh_pfor(const pos_type &p)
  {
    auto u = Lattice.toUnit_floor(p);
    #pragma omp for nowait
    for (int i = 0; i < nBlocks; ++i)
      compute_engine.evaluate_vgh(&einsplines[i], u[0], u[1], u[2],
                                  psi(i)->data(), grad(i)->data(), hess(i)->data(),
                                  psi(i).extent(0));
  }

  void print(std::ostream &os)
  {
    os << "SPO nBlocks=" << nBlocks << " firstBlock=" << firstBlock
       << " lastBlock=" << lastBlock << " nSplines=" << nSplines
       << " nSplinesPerBlock=" << nSplinesPerBlock << std::endl;
  }
};
}

#endif
