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
/** @file
 */
#ifndef QMCPLUSPLUS_EINSPLINE_SPO_HPP
#define QMCPLUSPLUS_EINSPLINE_SPO_HPP
#include <Utilities/Configuration.h>
#include <Utilities/NewTimer.h>
#include <Particle/ParticleSet.h>
#include "Devices.h"
#include <Numerics/Spline2/BsplineAllocator.hpp>
#include <Numerics/Spline2/MultiBsplineFuncs.hpp>
#include <Utilities/SIMD/allocator.hpp>
#include "Numerics/OhmmsPETE/OhmmsArray.h"
#include "QMCWaveFunctions/EinsplineSPOParams.h"
#include "QMCWaveFunctions/EinsplineSPODevice.hpp"
#include "QMCWaveFunctions/EinsplineSPODeviceImp.hpp"
#include "QMCWaveFunctions/SPOSetImp.h"
#include <iostream>

namespace qmcplusplus
{
template<Devices DT, typename T>
class EinsplineSPO : public SPOSetImp<DT>
{
public:
  // Global Type Aliases
  using QMCT         = QMCTraits;
  using PosType      = QMCT::PosType;
  using lattice_type = CrystalLattice<T, 3>;
  // Base SPOSetImp
  using BaseSPO = SPOSetImp<DT>;

  /// Timer
  NewTimer* timer;
  NewTimer* evalV_timer;

  /// default constructor
  EinsplineSPO()
  {
    std::cout << "EinsplineSPO() called\n";
    timer       = TimerManagerClass::get().createTimer("Single-Particle Orbitals", timer_level_fine);
    evalV_timer = TimerManagerClass::get().createTimer("Eval V", timer_level_fine);
  }

  EinsplineSPO(const EinsplineSPO& in) : einspline_spo_device(in.einspline_spo_device)
  {
    timer       = TimerManagerClass::get().createTimer("Single-Particle Orbitals", timer_level_fine);
    evalV_timer = TimerManagerClass::get().createTimer("Eval V", timer_level_fine);
  }

  // EinsplineSPO(const EinsplineSPO& in)
  //     : einspline_spo_device(in.enspline_spo_device)
  // {}

  /// disable assignment operator
  EinsplineSPO& operator=(const EinsplineSPO& in) = delete;

  /** "Fat" copy constructor
   * @param in EinsplineSPO
   * @param team_size number of members in a team
   * @param member_id id of this member in a team
   *
   * Create a view of the big object. A simple blocking & padding  method.
   * However this isn't the way kokkos does it.
   */
  EinsplineSPO(const EinsplineSPO& in, int team_size, int member_id)
      : einspline_spo_device(in.einspline_spo_device, team_size, member_id)
  {
    timer       = TimerManagerClass::get().createTimer("Single-Particle Orbitals", timer_level_fine);
    evalV_timer = TimerManagerClass::get().createTimer("Eval V", timer_level_fine);
  }

  /// destructors
  ~EinsplineSPO()
  {
    //Note the change in garbage collection here.  The reason for doing this is that by
    //changing einsplines to a view, it's more natural to work by reference than by raw pointer.
    // To maintain current interface, redoing the input types of allocate and destroy to call by references
    //  would need to be propagated all the way down.
    // However, since we've converted the large chunks of memory to views, garbage collection is
    // handled automatically.  Thus, setting the spline_type objects to empty views lets Kokkos handle the Garbage collection.

    //    for (int i = 0; i < nBlocks; ++i)
    //      myAllocator.destroy(einsplines(i));
  }


  // fix for general num_splines
  void set(int nx, int ny, int nz, int num_splines, int num_blocks, int splines_per_block, bool init_random = true)
  {
    einspline_spo_device.set(nx, ny, nz, num_splines, num_blocks, splines_per_block, init_random);
  }

  /** evaluate psi */
  inline void evaluate_v(const PosType& p)
  {
    ScopedTimer local_timer(timer);
    ScopedTimer another_local_timer(evalV_timer);
    einspline_spo_device.evaluate_v(p);
  }

  /** evaluate psi probably for synced walker */
  inline void evaluate_v_pfor(const PosType& p) { einspline_spo_device.evaluate_v_pfor(p); }

  /** evaluate psi, grad and lap */
  inline void evaluate_vgl(const PosType& p)
  {
    ScopedTimer local_timer(timer);
    einspline_spo_device.evaluate_vgl(p);
  }

  /** evaluate psi, grad and hess */
  void evaluate_vgh(const PosType& p)
  {
    ScopedTimer local_timer(timer);
    einspline_spo_device.evaluate_vgh(p);
  }

  void print(std::ostream& os) { os << einspline_spo_device; }

  void setLattice(const Tensor<T, 3>& lattice) { einspline_spo_device.setLattice(lattice); }

  const EinsplineSPOParams<T>& getParams() const { return einspline_spo_device.getParams(); }

  ///Access to elements in psi
  T getPsi(int ib, int n) { return einspline_spo_device.getPsi(ib, n); }

  T getGrad(int ib, int n, int m) { return einspline_spo_device.getGrad(ib, n, m); }

  T getHess(int ib, int n, int m) { return einspline_spo_device.getHess(ib, n, m); }

  EinsplineSPODeviceImp<DT, T> einspline_spo_device;

private:
};

// template<Devices DT, typename T>
// KOKKOS_INLINE_FUNCTION void EinsplineSPO<DT, T>::
//     operator()(const EvaluateVTag&, const team_v_serial_t& team) const
// {
// }

// template<Devices DT, typename T>
// KOKKOS_INLINE_FUNCTION void EinsplineSPO<DT, T>::
//     operator()(const EvaluateVTag&, const team_v_parallel_t& team) const
// {
// }

// template<typename T>
// KOKKOS_INLINE_FUNCTION void EinsplineSPO<Devices::KOKKOS, T>::
// operator()(const typename EvaluateVTag&,
// 	   const typename EinsplineSPO<Devices::KOKKOS, T>::team_v_serial_t& team) const
// {
//   int block               = team.league_rank();
//   auto u                  = Lattice.toUnit_floor(tmp_pos);
//   einsplines(block).coefs = einsplines(block).coefs_view.data();
//   compute_engine
//       .evaluate_v(team, &einsplines(block), u[0], u[1], u[2], psi(block).data(), psi(block).extent(0));
// }

// template<typename T>
// KOKKOS_INLINE_FUNCTION void EinsplineSPO<Devices::KOKKOS, T>::
// operator()(const typename EinsplineSPO<Devices::KOKKOS, T>::eEvaluateVTag&,
// 	   const typename EinsplineSPO<Devices::KOKKOS, T>::team_v_parallel_t& team) const
// {
//   int block               = team.league_rank();
//   auto u                  = Lattice.toUnit_floor(tmp_pos);
//   einsplines(block).coefs = einsplines(block).coefs_view.data();
//   compute_engine
//       .evaluate_v(team, &einsplines(block), u[0], u[1], u[2], psi(block).data(), psi(block).extent(0));
// }

} // namespace qmcplusplus

#endif
