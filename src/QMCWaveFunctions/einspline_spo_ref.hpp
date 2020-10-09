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
/** @file einspline_spo_ref.hpp
 */
#ifndef QMCPLUSPLUS_EINSPLINE_SPO_REF_HPP
#define QMCPLUSPLUS_EINSPLINE_SPO_REF_HPP
#include <Utilities/Configuration.h>
#include <Utilities/NewTimer.h>
#include <Particle/ParticleSet.h>
#include <Numerics/Spline2/MultiBsplineRef.hpp>
#include <CPU/SIMD/aligned_allocator.hpp>
#include "Numerics/OhmmsPETE/OhmmsArray.h"
#include "QMCWaveFunctions/SPOSet.h"
#include <iostream>

namespace miniqmcreference
{
using namespace qmcplusplus;

template<typename T>
struct einspline_spo_ref : public SPOSet
{
  /// define the einsplie data object type
  using spline_type     = typename bspline_traits<T, 3>::SplineType;
  using vContainer_type = aligned_vector<T>;
  using gContainer_type = VectorSoAContainer<T, 3>;
  using hContainer_type = VectorSoAContainer<T, 6>;
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

  aligned_vector<spline_type*> einsplines;
  aligned_vector<vContainer_type> psi;
  aligned_vector<gContainer_type> grad;
  aligned_vector<hContainer_type> hess;


  /// Timer
  NewTimer* timer;

  /// default constructor
  einspline_spo_ref() : nBlocks(0), nSplines(0), firstBlock(0), lastBlock(0), Owner(false)
  {
    timer = TimerManager.createTimer("Single-Particle Orbitals Ref", timer_level_fine);
  }
  /// disable copy constructor
  einspline_spo_ref(const einspline_spo_ref& in) = delete;
  /// disable copy operator
  einspline_spo_ref& operator=(const einspline_spo_ref& in) = delete;

  /** copy constructor
   * @param in einspline_spo_ref
   * @param team_size number of members in a team
   * @param member_id id of this member in a team
   *
   * Create a view of the big object. A simple blocking & padding  method.
   */
  einspline_spo_ref(const einspline_spo_ref& in, int team_size, int member_id) : Owner(false), Lattice(in.Lattice)
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
      einsplines[i] = in.einsplines[t];
    resize();
    timer = TimerManager.createTimer("Single-Particle Orbitals Ref", timer_level_fine);
  }

  /// destructors
  ~einspline_spo_ref()
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

  /// If not initialized previously, generate splines coeficients of \p num_splines
  /// divided into \p nblocks chunks each with a grid \p nx x \p ny x \p nz.
  /// If \p init_random is true, in each chunk, one orbital is fully randomized
  /// and others are tweaked based on it.
  void set(int nx, int ny, int nz, int num_splines, int nblocks, bool init_random = true)
  {
    // setting OrbitalSetSize to num_splines made artificial only in miniQMC
    OrbitalSetSize = num_splines;

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
          // Generate a orbital fully with fully randomized coefficients
          myrandom.generate_uniform(coef_data.data(), coef_data.size());
          // Generate different coefficients for each orbital by tweaking coef_data
          myAllocator.setCoefficientsForOrbitals(0, nSplinesPerBlock, coef_data, einsplines[i]);
        }
      }
    }
    resize();
  }

  /** evaluate psi */
  inline void evaluate_v(const ParticleSet& P, int iat)
  {
    ScopedTimer local_timer(timer);

    auto u = Lattice.toUnit_floor(P.activeR(iat));
    for (int i = 0; i < nBlocks; ++i)
      MultiBsplineEvalRef::evaluate_v(einsplines[i], u[0], u[1], u[2], psi[i].data(), nSplinesPerBlock);
  }

  inline void evaluate(const ParticleSet& P, int iat, ValueVector_t& psi_v)
  {
    evaluate_v(P, iat);

    for (int i = 0; i < nBlocks; ++i)
    {
      // in real simulation, phase needs to be applied. Here just fake computation
      const int first = i * nBlocks;
      std::copy_n(psi[i].data(), std::min((i + 1) * nSplinesPerBlock, OrbitalSetSize) - first, psi_v.data() + first);
    }
  }

  /** evaluate psi, grad and lap */
  inline void evaluate_vgl(const ParticleSet& P, int iat)
  {
    auto u = Lattice.toUnit_floor(P.activeR(iat));
    for (int i = 0; i < nBlocks; ++i)
      MultiBsplineEvalRef::evaluate_vgl(einsplines[i], u[0], u[1], u[2], psi[i].data(), grad[i].data(), hess[i].data(),
                                        nSplinesPerBlock);
  }

  /** evaluate psi, grad and hess */
  inline void evaluate_vgh(const ParticleSet& P, int iat)
  {
    ScopedTimer local_timer(timer);

    auto u = Lattice.toUnit_floor(P.activeR(iat));
    for (int i = 0; i < nBlocks; ++i)
      MultiBsplineEvalRef::evaluate_vgh(einsplines[i], u[0], u[1], u[2], psi[i].data(), grad[i].data(), hess[i].data(),
                                        nSplinesPerBlock);
  }

  inline void evaluate(const ParticleSet& P,
                       int iat,
                       ValueVector_t& psi_v,
                       GradVector_t& dpsi_v,
                       ValueVector_t& d2psi_v)
  {
    evaluate_vgh(P, iat);

    for (int i = 0; i < nBlocks; ++i)
    {
      // in real simulation, phase needs to be applied. Here just fake computation
      const int first = i * nBlocks;
      for (int j = first; j < std::min((i + 1) * nSplinesPerBlock, OrbitalSetSize); j++)
      {
        psi_v[j]   = psi[i][j - first];
        dpsi_v[j]  = grad[i][j - first];
        d2psi_v[j] = hess[i].data(0)[j - first];
      }
    }
  }

  void print(std::ostream& os)
  {
    os << "SPO nBlocks=" << nBlocks << " firstBlock=" << firstBlock << " lastBlock=" << lastBlock
       << " nSplines=" << nSplines << " nSplinesPerBlock=" << nSplinesPerBlock << std::endl;
  }
};
} // namespace miniqmcreference

#endif
