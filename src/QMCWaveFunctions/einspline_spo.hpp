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
#include <Configuration.h>
#include <Particle/ParticleSet.h>
#include <spline2/bspline_allocator.hpp>
#include <spline2/MultiBspline.hpp>
#include <simd/allocator.hpp>
#include "OhmmsPETE/OhmmsArray.h"
#include "OMP_target_test/OMPTinyVector.h"
#include <iostream>

namespace qmcplusplus
{
template <typename T, typename compute_engine_type = MultiBspline<T> >
struct einspline_spo
{
  /// define the einsplie data object type
  using self_type = einspline_spo<T, compute_engine_type>;
  using spline_type = typename bspline_traits<T, 3>::SplineType;
  using pos_type        = TinyVector<T, 3>;
  using vContainer_type = aligned_vector<T>;
  using gContainer_type = VectorSoaContainer<T, 3>;
  using hContainer_type = VectorSoaContainer<T, 6>;
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

  std::vector<spline_type *> einsplines;
  std::vector<vContainer_type> psi;
  std::vector<gContainer_type> grad;
  std::vector<hContainer_type> hess;

  // for shadows
  std::vector<T*> psi_shadows;
  std::vector<T*> grad_shadows;
  std::vector<T*> hess_shadows;
  std::vector<OMPTinyVector<T, 3> > u_shadows;

  /// default constructor
  einspline_spo()
      : nBlocks(0), nSplines(0), firstBlock(0), lastBlock(0), Owner(false)
  {
  }
  /// disable copy constructor
  einspline_spo(const einspline_spo &in) = delete;
  /// disable copy operator
  einspline_spo &operator=(const einspline_spo &in) = delete;

  /** copy constructor
   * @param in einspline_spo
   * @param ncrews number of crews of a team
   * @param crewID id of this crew in a team
   *
   * Create a view of the big object. A simple blocking & padding  method.
   */
  einspline_spo(einspline_spo &in, int ncrews, int crewID)
      : Owner(false), Lattice(in.Lattice)
  {
    nSplines         = in.nSplines;
    nSplinesPerBlock = in.nSplinesPerBlock;
    nBlocks          = (in.nBlocks + ncrews - 1) / ncrews;
    firstBlock       = nBlocks * crewID;
    lastBlock        = std::min(in.nBlocks, nBlocks * (crewID + 1));
    nBlocks          = lastBlock - firstBlock;
    einsplines.resize(nBlocks);
    for (int i = 0, t = firstBlock; i < nBlocks; ++i, ++t)
      einsplines[i] = in.einsplines[t];
    resize();
  }

  /// destructors
  ~einspline_spo()
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
  void set(int nx, int ny, int nz, int num_splines, int nblocks,
           bool init_random = true)
  {
    nSplines         = num_splines;
    nBlocks          = nblocks;
    nSplinesPerBlock = num_splines / nblocks;
    firstBlock       = 0;
    lastBlock        = nBlocks;
    if (einsplines.empty())
    {
      Owner = true;
      TinyVector<int, 3> ng(nx, ny, nz);
      pos_type start(0);
      pos_type end(1);
      einsplines.resize(nBlocks);
      RandomGenerator<T> myrandom(11);
      Array<T, 3> data(nx, ny, nz);
      std::fill(data.begin(), data.end(), T());
      myrandom.generate_uniform(data.data(), data.size());
      for (int i = 0; i < nBlocks; ++i)
      {
        einsplines[i] = myAllocator.createMultiBspline(T(0), start, end, ng, PERIODIC, nSplinesPerBlock);
        if (init_random)
          for (int j = 0; j < nSplinesPerBlock; ++j)
            myAllocator.set(data.data(), einsplines[i], j);
      }
    }
    resize();
  }

  /** evaluate psi */
  inline void evaluate_v(const pos_type &p)
  {
    auto u = Lattice.toUnit(p);
    for (int i = 0; i < nBlocks; ++i)
      compute_engine.evaluate_v(einsplines[i], u[0], u[1], u[2], psi[i].data(), nSplinesPerBlock);
  }

  /** evaluate psi */
  inline void evaluate_v_pfor(const pos_type &p)
  {
    auto u = Lattice.toUnit(p);
    #pragma omp for nowait
    for (int i = 0; i < nBlocks; ++i)
      compute_engine.evaluate_v(einsplines[i], u[0], u[1], u[2], psi[i].data(), nSplinesPerBlock);
  }

  /** evaluate psi, grad and lap */
  inline void evaluate_vgl(const pos_type &p)
  {
    auto u = Lattice.toUnit(p);
    for (int i = 0; i < nBlocks; ++i)
      compute_engine.evaluate_vgl(einsplines[i], u[0], u[1], u[2],
                                  psi[i].data(), grad[i].data(), hess[i].data(),
                                  nSplinesPerBlock);
  }

  /** evaluate psi, grad and lap */
  inline void evaluate_vgl_pfor(const pos_type &p)
  {
    auto u = Lattice.toUnit(p);
    #pragma omp for nowait
    for (int i = 0; i < nBlocks; ++i)
      compute_engine.evaluate_vgl(einsplines[i], u[0], u[1], u[2],
                                  psi[i].data(), grad[i].data(), hess[i].data(),
                                  nSplinesPerBlock);
  }

  /** evaluate psi, grad and hess */
  inline void evaluate_vgh(const pos_type &p)
  {
    auto u = Lattice.toUnit(p);
    for (int i = 0; i < nBlocks; ++i)
      compute_engine.evaluate_vgh(einsplines[i], u[0], u[1], u[2],
                                  psi[i].data(), grad[i].data(), hess[i].data(),
                                  nSplinesPerBlock);
  }

  /** evaluate psi, grad and hess */
  inline void evaluate_vgh_pfor(const pos_type &p)
  {
    auto u = Lattice.toUnit(p);
    #pragma omp for nowait
    for (int i = 0; i < nBlocks; ++i)
      compute_engine.evaluate_vgh(einsplines[i], u[0], u[1], u[2],
                                  psi[i].data(), grad[i].data(), hess[i].data(),
                                  nSplinesPerBlock);
  }

  /** evaluate psi, grad and hess */
  inline void evaluate_multi_vgh(const std::vector<pos_type> &p, std::vector<self_type *> &shadows)
  {
    const size_t nw = p.size();
    psi_shadows.resize(nw*nBlocks);
    grad_shadows.resize(nw*nBlocks);
    hess_shadows.resize(nw*nBlocks);
    u_shadows.resize(nw);

    for(size_t iw = 0; iw < nw; iw++)
    {
      u_shadows[iw] = Lattice.toUnit(p[iw]);
      auto &shadow = *shadows[iw];
      for (int i = 0; i < nBlocks; ++i)
      {
        psi_shadows[iw*nBlocks+i] = shadow.psi[i].data();
        grad_shadows[iw*nBlocks+i] = shadow.grad[i].data();
        hess_shadows[iw*nBlocks+i] = shadow.hess[i].data();
      }
    }

    for(size_t iw = 0; iw < nw; iw++)
      for (size_t i = 0; i < nBlocks; ++i)
        compute_engine.evaluate_vgh(einsplines[i], u_shadows[iw][0], u_shadows[iw][1], u_shadows[iw][2],
                                    psi_shadows[iw*nBlocks+i], grad_shadows[iw*nBlocks+i],
                                    hess_shadows[iw*nBlocks+i], nSplinesPerBlock);
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
