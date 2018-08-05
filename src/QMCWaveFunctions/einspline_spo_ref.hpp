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
<<<<<<< HEAD
/** @file einspline_spo.hpp
=======
/** @file einspline_spo_ref.hpp
>>>>>>> develop
 */
#ifndef QMCPLUSPLUS_EINSPLINE_SPO_REF_HPP
#define QMCPLUSPLUS_EINSPLINE_SPO_REF_HPP
#include <Utilities/Configuration.h>
#include <Utilities/NewTimer.h>
#include <Particle/ParticleSet.h>
#include <Numerics/Spline2/bspline_allocator.hpp>
<<<<<<< HEAD
#include <Numerics/Spline2/MultiBspline.hpp>
#include <Utilities/SIMD/allocator.hpp>
#include "Numerics/OhmmsPETE/OhmmsArray.h"
#include <iostream>

namespace qmcplusplus
{
template <typename T, typename compute_engine_type = MultiBspline<T> >
struct einspline_spo_ref
{
  /// define the einsplie data object type
  using spline_type = typename bspline_traits<T, 3>::SplineType;
  using pos_type        = TinyVector<T, 3>;
  using vContainer_type = Kokkos::View<T*>;
  using gContainer_type = Kokkos::View<T*[3],Kokkos::LayoutLeft>;
  using hContainer_type = Kokkos::View<T*[6],Kokkos::LayoutLeft>;
=======
#include <Numerics/Spline2/MultiBsplineRef.hpp>
#include <Utilities/SIMD/allocator.hpp>
#include "Numerics/OhmmsPETE/OhmmsArray.h"
#include "QMCWaveFunctions/SPOSet.h"
#include <iostream>

namespace miniqmcreference
{

using namespace qmcplusplus;

template <typename T>
struct einspline_spo_ref : public SPOSet
{
  /// define the einsplie data object type
  using spline_type = typename bspline_traits<T, 3>::SplineType;
  using vContainer_type = aligned_vector<T>;
  using gContainer_type = VectorSoAContainer<T, 3>;
  using hContainer_type = VectorSoAContainer<T, 6>;
>>>>>>> develop
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
<<<<<<< HEAD
  compute_engine_type compute_engine;

  Kokkos::View<spline_type *> einsplines;
  Kokkos::View<vContainer_type*> psi;
  Kokkos::View<gContainer_type*> grad;
  Kokkos::View<hContainer_type*> hess;
=======
  MultiBsplineRef<T> compute_engine;

  aligned_vector<spline_type *> einsplines;
  aligned_vector<vContainer_type> psi;
  aligned_vector<gContainer_type> grad;
  aligned_vector<hContainer_type> hess;
>>>>>>> develop


  /// Timer
  NewTimer *timer;

  /// default constructor
  einspline_spo_ref()
      : nBlocks(0), nSplines(0), firstBlock(0), lastBlock(0), Owner(false)
  {
<<<<<<< HEAD
    timer = TimerManager.createTimer("Single-Particle Orbitals", timer_level_coarse);
=======
    timer = TimerManager.createTimer("Single-Particle Orbitals Ref", timer_level_fine);
>>>>>>> develop
  }
  /// disable copy constructor
  einspline_spo_ref(const einspline_spo_ref &in) = delete;
  /// disable copy operator
  einspline_spo_ref &operator=(const einspline_spo_ref &in) = delete;

  /** copy constructor
<<<<<<< HEAD
   * @param in einspline_spo
=======
   * @param in einspline_spo_ref
>>>>>>> develop
   * @param team_size number of members in a team
   * @param member_id id of this member in a team
   *
   * Create a view of the big object. A simple blocking & padding  method.
   */
<<<<<<< HEAD
  einspline_spo_ref(einspline_spo_ref &in, int team_size, int member_id)
=======
  einspline_spo_ref(const einspline_spo_ref &in, int team_size, int member_id)
>>>>>>> develop
      : Owner(false), Lattice(in.Lattice)
  {
    nSplines         = in.nSplines;
    nSplinesPerBlock = in.nSplinesPerBlock;
    nBlocks          = (in.nBlocks + team_size - 1) / team_size;
    firstBlock       = nBlocks * member_id;
    lastBlock        = std::min(in.nBlocks, nBlocks * (member_id + 1));
    nBlocks          = lastBlock - firstBlock;
<<<<<<< HEAD
   // einsplines.resize(nBlocks);
    einsplines       = Kokkos::View<spline_type*>("einsplines",nBlocks);
    for (int i = 0, t = firstBlock; i < nBlocks; ++i, ++t)
      einsplines(i) = in.einsplines(t);
    resize();
    timer = TimerManager.createTimer("Single-Particle Orbitals", timer_level_coarse);
=======
    einsplines.resize(nBlocks);
    for (int i = 0, t = firstBlock; i < nBlocks; ++i, ++t)
      einsplines[i] = in.einsplines[t];
    resize();
    timer = TimerManager.createTimer("Single-Particle Orbitals Ref", timer_level_fine);
>>>>>>> develop
  }

  /// destructors
  ~einspline_spo_ref()
  {
<<<<<<< HEAD
    //Note the change in garbage collection here.  The reason for doing this is that by
    //changing einsplines to a view, it's more natural to work by reference than by raw pointer.  
    // To maintain current interface, redoing the input types of allocate and destroy to call by references
    //  would need to be propagated all the way down.  
    // However, since we've converted the large chunks of memory to views, garbage collection is
    // handled automatically.  Thus, setting the spline_type objects to empty views lets Kokkos handle the Garbage collection.

    if (Owner)
      einsplines = Kokkos::View<spline_type*>();
      
  //    for (int i = 0; i < nBlocks; ++i)
  //      myAllocator.destroy(einsplines(i));
=======
    if (Owner)
      for (int i = 0; i < nBlocks; ++i)
        myAllocator.destroy(einsplines[i]);
>>>>>>> develop
  }

  /// resize the containers
  void resize()
  {
<<<<<<< HEAD
//    psi.resize(nBlocks);
//    grad.resize(nBlocks);
//    hess.resize(nBlocks);

    psi = Kokkos::View<vContainer_type*>("Psi",nBlocks);
    grad = Kokkos::View<gContainer_type*>("Grad",nBlocks);
    hess = Kokkos::View<hContainer_type*>("Hess",nBlocks);

    for (int i = 0; i < nBlocks; ++i)
    {
      //psi[i].resize(nSplinesPerBlock);
      //grad[i].resize(nSplinesPerBlock);
      //hess[i].resize(nSplinesPerBlock);
     
      //Using the "view-of-views" placement-new construct.
      new (&psi(i))  vContainer_type("psi_i",nSplinesPerBlock);
      new (&grad(i)) gContainer_type("grad_i",nSplinesPerBlock);
      new (&hess(i)) hContainer_type("hess_i",nSplinesPerBlock);
=======
    psi.resize(nBlocks);
    grad.resize(nBlocks);
    hess.resize(nBlocks);
    for (int i = 0; i < nBlocks; ++i)
    {
      psi[i].resize(nSplinesPerBlock);
      grad[i].resize(nSplinesPerBlock);
      hess[i].resize(nSplinesPerBlock);
>>>>>>> develop
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
<<<<<<< HEAD
    if (! einsplines.extent(0))
    {
      Owner = true;
      TinyVector<int, 3> ng(nx, ny, nz);
      pos_type start(0);
      pos_type end(1);
      
//    einsplines.resize(nBlocks);
      einsplines = Kokkos::View<spline_type*>("einsplines",nBlocks);

      RandomGenerator<T> myrandom(11);
      //Array<T, 3> coef_data(nx+3, ny+3, nz+3);
      Kokkos::View<T***> coef_data("coef_data",nx+3,ny+3,nz+3);

      for (int i = 0; i < nBlocks; ++i)
      {
        einsplines(i) = *myAllocator.createMultiBspline(T(0), start, end, ng, PERIODIC, nSplinesPerBlock);
        if (init_random) {
          for (int j = 0; j < nSplinesPerBlock; ++j) {
            // Generate different coefficients for each orbital
            myrandom.generate_uniform(coef_data.data(), coef_data.extent(0));
            myAllocator.setCoefficientsForOneOrbital(j, coef_data, &einsplines(i));
=======
    if (einsplines.empty())
    {
      Owner = true;
      TinyVector<int, 3> ng(nx, ny, nz);
      PosType start(0);
      PosType end(1);
      einsplines.resize(nBlocks);
      RandomGenerator<T> myrandom(11);
      Array<T, 3> coef_data(nx+3, ny+3, nz+3);
      for (int i = 0; i < nBlocks; ++i)
      {
        einsplines[i] = myAllocator.createMultiBspline(T(0), start, end, ng, PERIODIC, nSplinesPerBlock);
        if (init_random) {
          for (int j = 0; j < nSplinesPerBlock; ++j) {
            // Generate different coefficients for each orbital
            myrandom.generate_uniform(coef_data.data(), coef_data.size());
            myAllocator.setCoefficientsForOneOrbital(j, coef_data, einsplines[i]);
>>>>>>> develop
          }
        }
      }
    }
    resize();
  }

  /** evaluate psi */
<<<<<<< HEAD
  inline void evaluate_v(const pos_type &p)
=======
  inline void evaluate_v(const PosType &p)
>>>>>>> develop
  {
    ScopedTimer local_timer(timer);

    auto u = Lattice.toUnit_floor(p);
    for (int i = 0; i < nBlocks; ++i)
<<<<<<< HEAD
      compute_engine.evaluate_v(&einsplines(i), u[0], u[1], u[2], psi(i).data(), nSplinesPerBlock);
  }

  /** evaluate psi */
  inline void evaluate_v_pfor(const pos_type &p)
=======
      compute_engine.evaluate_v(einsplines[i], u[0], u[1], u[2], psi[i].data(), nSplinesPerBlock);
  }

  /** evaluate psi */
  inline void evaluate_v_pfor(const PosType &p)
>>>>>>> develop
  {
    auto u = Lattice.toUnit_floor(p);
    #pragma omp for nowait
    for (int i = 0; i < nBlocks; ++i)
<<<<<<< HEAD
      compute_engine.evaluate_v(&einsplines(i), u[0], u[1], u[2], psi(i).data(), nSplinesPerBlock);
  }

  /** evaluate psi, grad and lap */
  inline void evaluate_vgl(const pos_type &p)
  {
    auto u = Lattice.toUnit_floor(p);
    for (int i = 0; i < nBlocks; ++i)
      compute_engine.evaluate_vgl(&einsplines(i), u[0], u[1], u[2],
                                  psi(i).data(), grad(i).data(), hess(i).data(),
=======
      compute_engine.evaluate_v(einsplines[i], u[0], u[1], u[2], psi[i].data(), nSplinesPerBlock);
  }

  /** evaluate psi, grad and lap */
  inline void evaluate_vgl(const PosType &p)
  {
    auto u = Lattice.toUnit_floor(p);
    for (int i = 0; i < nBlocks; ++i)
      compute_engine.evaluate_vgl(einsplines[i], u[0], u[1], u[2],
                                  psi[i].data(), grad[i].data(), hess[i].data(),
>>>>>>> develop
                                  nSplinesPerBlock);
  }

  /** evaluate psi, grad and lap */
<<<<<<< HEAD
  inline void evaluate_vgl_pfor(const pos_type &p)
=======
  inline void evaluate_vgl_pfor(const PosType &p)
>>>>>>> develop
  {
    auto u = Lattice.toUnit_floor(p);
    #pragma omp for nowait
    for (int i = 0; i < nBlocks; ++i)
<<<<<<< HEAD
      compute_engine.evaluate_vgl(&einsplines(i), u[0], u[1], u[2],
                                  psi(i).data(), grad(i).data(), hess(i).data(),
=======
      compute_engine.evaluate_vgl(einsplines[i], u[0], u[1], u[2],
                                  psi[i].data(), grad[i].data(), hess[i].data(),
>>>>>>> develop
                                  nSplinesPerBlock);
  }

  /** evaluate psi, grad and hess */
<<<<<<< HEAD
  inline void evaluate_vgh(const pos_type &p)
=======
  inline void evaluate_vgh(const PosType &p)
>>>>>>> develop
  {
    ScopedTimer local_timer(timer);

    auto u = Lattice.toUnit_floor(p);
    for (int i = 0; i < nBlocks; ++i)
<<<<<<< HEAD
      compute_engine.evaluate_vgh(&einsplines(i), u[0], u[1], u[2],
                                  psi(i).data(), grad(i).data(), hess(i).data(),
=======
      compute_engine.evaluate_vgh(einsplines[i], u[0], u[1], u[2],
                                  psi[i].data(), grad[i].data(), hess[i].data(),
>>>>>>> develop
                                  nSplinesPerBlock);
  }

  /** evaluate psi, grad and hess */
<<<<<<< HEAD
  inline void evaluate_vgh_pfor(const pos_type &p)
=======
  inline void evaluate_vgh_pfor(const PosType &p)
>>>>>>> develop
  {
    auto u = Lattice.toUnit_floor(p);
    #pragma omp for nowait
    for (int i = 0; i < nBlocks; ++i)
<<<<<<< HEAD
      compute_engine.evaluate_vgh(&einsplines(i), u[0], u[1], u[2],
                                  psi(i).data(), grad(i).data(), hess(i).data(),
=======
      compute_engine.evaluate_vgh(einsplines[i], u[0], u[1], u[2],
                                  psi[i].data(), grad[i].data(), hess[i].data(),
>>>>>>> develop
                                  nSplinesPerBlock);
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
