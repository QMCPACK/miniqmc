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
/** @file EinsplineSPO_ref.hpp
 */
#ifndef QMCPLUSPLUS_EINSPLINE_SPO_REF_HPP
#define QMCPLUSPLUS_EINSPLINE_SPO_REF_HPP
#include <Utilities/Configuration.h>
#include <Utilities/NewTimer.h>
#include <Particle/ParticleSet.h>
#include <Numerics/Spline2/BsplineAllocator.hpp>
#include <Numerics/Spline2/MultiBsplineRef.hpp>
#include <Utilities/SIMD/allocator.hpp>
#include "Utilities/RandomGenerator.h"
#include "Numerics/OhmmsPETE/OhmmsArray.h"
#include "QMCWaveFunctions/SPOSetImp.h"


namespace miniqmcreference
{
using namespace qmcplusplus;

template<typename T>
struct EinsplineSPO_ref : public qmcplusplus::SPOSetImp<Devices::CPU>
{
  /// define the einsplie data object type
  using spline_type     = typename bspline_traits<Devices::CPU, T, 3>::SplineType;
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
  einspline::Allocator<Devices::CPU> myAllocator;
  /// compute engine
  MultiBsplineRef<T> compute_engine;

  aligned_vector<spline_type*> einsplines;
  aligned_vector<vContainer_type> psi;
  aligned_vector<gContainer_type> grad;
  aligned_vector<hContainer_type> hess;


  /// Timer
  NewTimer* timer;

  /// default constructor
  EinsplineSPO_ref() : nBlocks(0), nSplines(0), firstBlock(0), lastBlock(0), Owner(false)
  {
    timer = TimerManagerClass::get().createTimer("Single-Particle Orbitals Ref", timer_level_fine);
  }

  /** Copy constructor
   * Needed for Kokkos since it needs pass by copy
   * Of course you'd also need this if you were doing
   * MPI in a straight forward way
   */
  EinsplineSPO_ref(const EinsplineSPO_ref& in) : Owner(false), Lattice(in.Lattice)
  {
    nSplines         = in.nSplines;
    nSplinesPerBlock = in.nSplinesPerBlock;
    nBlocks          = in.nBlocks;
    firstBlock       = 0;
    lastBlock        = in.nBlocks;
    einsplines.resize(nBlocks);
    for (int i = 0, t = firstBlock; i < nBlocks; ++i, ++t)
      einsplines[i] = in.einsplines[t];
    resize();
    timer = TimerManagerClass::get().createTimer("Single-Particle Orbitals Ref", timer_level_fine);
  }

  EinsplineSPO_ref& operator=(const EinsplineSPO_ref& in) = delete;

  /** "Fat" copy constructor
   * @param in EinsplineSPO_ref
   * @param team_size number of members in a team
   * @param member_id id of this member in a team
   *
   * Create a view of the big object. A simple blocking & padding  method.
   */
  EinsplineSPO_ref(const EinsplineSPO_ref& in, int team_size, int member_id) : Owner(false), Lattice(in.Lattice)
  {
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
    timer = TimerManagerClass::get().createTimer("Single-Particle Orbitals Ref", timer_level_fine);
  }

  /// destructors
  ~EinsplineSPO_ref()
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
      std::cout << "Initializing Reference Spline Coefficients with nBlocks: " << nBlocks
                << " and nSplinesPerblock : " << nSplinesPerBlock << '\n';
      for (int i = 0; i < nBlocks; ++i)
      {
        myAllocator.createMultiBspline(einsplines[i], T(0), start, end, ng, PERIODIC, nSplinesPerBlock);
        if (init_random)
        {
          for (int j = 0; j < nSplinesPerBlock; ++j)
          {
            // Generate different coefficients for each orbital
            myrandom.generate_uniform(coef_data.data(), coef_data.size());
            myAllocator.setCoefficientsForOneOrbital(j, coef_data, einsplines[i]);
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

    auto u = Lattice.toUnit_floor(p);
    for (int i = 0; i < nBlocks; ++i)
      compute_engine.evaluate_v(einsplines[i], u[0], u[1], u[2], psi[i].data(), nSplinesPerBlock);
  }

  /** evaluate psi */
  inline void evaluate_v_pfor(const PosType& p)
  {
    auto u = Lattice.toUnit_floor(p);
#pragma omp for nowait
    for (int i = 0; i < nBlocks; ++i)
      compute_engine.evaluate_v(einsplines[i], u[0], u[1], u[2], psi[i].data(), nSplinesPerBlock);
  }

  /** evaluate psi, grad and lap */
  inline void evaluate_vgl(const PosType& p)
  {
    auto u = Lattice.toUnit_floor(p);
    for (int i = 0; i < nBlocks; ++i)
      compute_engine.evaluate_vgl(einsplines[i],
                                  u[0],
                                  u[1],
                                  u[2],
                                  psi[i].data(),
                                  grad[i].data(),
                                  hess[i].data(),
                                  nSplinesPerBlock);
  }

  /** evaluate psi, grad and lap */
  inline void evaluate_vgl_pfor(const PosType& p)
  {
    auto u = Lattice.toUnit_floor(p);
#pragma omp for nowait
    for (int i = 0; i < nBlocks; ++i)
      compute_engine.evaluate_vgl(einsplines[i],
                                  u[0],
                                  u[1],
                                  u[2],
                                  psi[i].data(),
                                  grad[i].data(),
                                  hess[i].data(),
                                  nSplinesPerBlock);
  }

  /** evaluate psi, grad and hess */
  inline void evaluate_vgh(const PosType& p)
  {
    ScopedTimer local_timer(timer);
    auto u = Lattice.toUnit_floor(p);
    for (int i = 0; i < nBlocks; ++i)
    {
      compute_engine.evaluate_vgh(einsplines[i],
                                  u[0],
                                  u[1],
                                  u[2],
                                  psi[i].data(),
                                  grad[i].data(),
                                  hess[i].data(),
                                  nSplinesPerBlock);
    }
  }

  /** evaluate psi, grad and hess */
  inline void evaluate_vgh_pfor(const PosType& p)
  {
    auto u = Lattice.toUnit_floor(p);
#pragma omp for nowait
    for (int i = 0; i < nBlocks; ++i)
      compute_engine.evaluate_vgh(einsplines[i],
                                  u[0],
                                  u[1],
                                  u[2],
                                  psi[i].data(),
                                  grad[i].data(),
                                  hess[i].data(),
                                  nSplinesPerBlock);
  }

  T getGrad(int ib, int n, int m) { return grad[ib].data(m)[n]; }

  T getHess(int ib, int n, int m) { return hess[ib].data(m)[n]; }

  void print(std::ostream& os)
  {
    os << "SPO nBlocks=" << nBlocks << " firstBlock=" << firstBlock << " lastBlock=" << lastBlock
       << " nSplines=" << nSplines << " nSplinesPerBlock=" << nSplinesPerBlock << std::endl;
  }
};

} // namespace miniqmcreference

#endif
