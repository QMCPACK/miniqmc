////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2018 QMCPACK developers.
//
// File developed by:
// Peter Doak, doakpw@ornl.gov, Oak Ridge National Lab
//
// File created by:
// Peter Doak, doakpw@ornl.gov, Oak Ridge National Lab
////////////////////////////////////////////////////////////////////////////////
// -*- C++ -*-

/**
 * @file
 * @brief Kokkos implementation of EinsplineSPO
 */

#ifndef QMCPLUSPLUS_EINSPLINE_SPO_DEVICE_IMP_KOKKOS_H
#define QMCPLUSPLUS_EINSPLINE_SPO_DEVICE_IMP_KOKKOS_H

#include "clean_inlining.h"
#include <impl/Kokkos_Timer.hpp>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include "Utilities/Configuration.h"
#include "Utilities/RandomGenerator.h"
#include "QMCWaveFunctions/EinsplineSPOParams.h"
#include "QMCWaveFunctions/EinsplineSPODevice.hpp"
#include "QMCWaveFunctions/EinsplineSPODeviceImp.hpp"
#include "Numerics/Spline2/bspline_traits.hpp"
#include "Numerics/Spline2/bspline_allocator.hpp"
#include "Numerics/Spline2/MultiBsplineFuncs.hpp"
#ifdef KOKKOS_ENABLE_CUDA
#include "cublas_v2.h"
#include "cusolverDn.h"
#endif

namespace qmcplusplus
{

template<typename T>
class EinsplineSPODeviceImp<Devices::KOKKOS, T>
  : public EinsplineSPODevice<EinsplineSPODeviceImp<Devices::KOKKOS, T>,T>
{
public:
  static constexpr Devices DT = Devices::KOKKOS;
  struct EvaluateVGHTag
  {};
  struct EvaluateVTag
  {};
  using QMCT = QMCTraits;
  EinsplineSPOParams<T> esp;
  using spline_type = typename bspline_traits<DT, T, 3>::SplineType;

  typedef Kokkos::TeamPolicy<Kokkos::Serial, EvaluateVGHTag> policy_vgh_serial_t;
  typedef Kokkos::TeamPolicy<EvaluateVGHTag> policy_vgh_parallel_t;
  typedef Kokkos::TeamPolicy<Kokkos::Serial, EvaluateVTag> policy_v_serial_t;
  typedef Kokkos::TeamPolicy<EvaluateVTag> policy_v_parallel_t;

  typedef typename policy_vgh_serial_t::member_type team_vgh_serial_t;
  typedef typename policy_vgh_parallel_t::member_type team_vgh_parallel_t;
  typedef typename policy_v_serial_t::member_type team_v_serial_t;
  typedef typename policy_v_parallel_t::member_type team_v_parallel_t;

  using vContainer_type = Kokkos::View<T*>;
  using gContainer_type = Kokkos::View<T * [3], Kokkos::LayoutLeft>;
  using hContainer_type = Kokkos::View<T * [6], Kokkos::LayoutLeft>;
  using lattice_type    = CrystalLattice<T, 3>;

  //using einspline_type = spline_type*;
  Kokkos::View<spline_type*> einsplines;
  Kokkos::View<vContainer_type*> psi;
  Kokkos::View<gContainer_type*> grad;
  Kokkos::View<hContainer_type*> hess;

  /// use allocator
  einspline::Allocator<DT> myAllocator;
  /// Compute engine
  MultiBsplineFuncs<DT, T> compute_engine;
  //Temporary position for communicating within Kokkos parallel sections.
  QMCT::PosType tmp_pos;
  NewTimer* timer;
  /// define the einsplie data object type

public:
  EinsplineSPODeviceImp()
    : tmp_pos(0)
  {
    esp.nSplinesSerialThreshold_V = 512;
    esp.nSplinesSerialThreshold_VGH = 128;
    timer = TimerManagerClass::get().createTimer("EinsplineSPODeviceImp<KOKKOS>", timer_level_fine);
  }
  
  //Copy Constructor only supports KOKKOS to KOKKOS
  EinsplineSPODeviceImp(const EinsplineSPODevice<EinsplineSPODeviceImp<Devices::KOKKOS, T>, T>& in,
                        int team_size,
                        int member_id)
    : EinsplineSPODevice<EinsplineSPODeviceImp<Devices::KOKKOS, T>, T>(in, team_size, member_id)
  {
    timer = TimerManagerClass::get().createTimer("EinsplineSPODeviceImp<KOKKOS>", timer_level_fine);
    const EinsplineSPOParams<T>& inesp = in.getParams();
    esp.nSplinesSerialThreshold_V   = inesp.nSplinesSerialThreshold_V;
    esp.nSplinesSerialThreshold_VGH = inesp.nSplinesSerialThreshold_VGH;
    esp.nSplines                    = inesp.nSplines;
    esp.nSplinesPerBlock            = inesp.nSplinesPerBlock;
    esp.nBlocks                     = (inesp.nBlocks + team_size - 1) / team_size;
    esp.firstBlock                  = esp.nBlocks * member_id;
    esp.lastBlock                   = std::min(inesp.nBlocks, esp.nBlocks * (member_id + 1));
    esp.nBlocks                     = esp.lastBlock - esp.firstBlock;
    einsplines                  = Kokkos::View<spline_type*>("einsplines", esp.nBlocks);
    for (int i = 0, t = esp.firstBlock; i < esp.nBlocks; ++i, ++t)
      //KOKKOS people take note, is this ok?
      einsplines(i)= *static_cast<spline_type*>(in.getEinspline(t));
    resize();
  }

  /// resize the containers
  void resize()
  {
    //    psi.resize(nBlocks);
    //    grad.resize(nBlocks);
    //    hess.resize(nBlocks);

    psi  = Kokkos::View<vContainer_type*>("Psi", esp.nBlocks);
    grad = Kokkos::View<gContainer_type*>("Grad", esp.nBlocks);
    hess = Kokkos::View<hContainer_type*>("Hess", esp.nBlocks);

    for (int i = 0; i < esp.nBlocks; ++i)
    {
      //psi[i].resize(nSplinesPerBlock);
      //grad[i].resize(nSplinesPerBlock);
      //hess[i].resize(nSplinesPerBlock);

      //Using the "view-of-views" placement-new construct.
      new (&psi(i)) vContainer_type("psi_i", esp.nSplinesPerBlock);
      new (&grad(i)) gContainer_type("grad_i", esp.nSplinesPerBlock);
      new (&hess(i)) hContainer_type("hess_i", esp.nSplinesPerBlock);
    }
  }

  ~EinsplineSPODeviceImp()
  {
    if (!esp.is_copy)
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
  }

    void set_i(int nx, int ny, int nz, int num_splines, int nblocks, int splines_per_block, bool init_random = true)
  {
    esp.nSplines         = num_splines;
    esp.nBlocks          = nblocks;
    esp.nSplinesPerBlock = splines_per_block;
    if ( num_splines > splines_per_block * nBlocks )
	throw std::runtime_error("splines_per_block * nblocks < num_splines");
    esp.firstBlock       = 0;
    esp.lastBlock        = nblocks;
    if (einsplines.extent(0) == 0)
    {
      esp.Owner = true;
      TinyVector<int, 3> ng(nx, ny, nz);
      QMCT::PosType start(0);
      QMCT::PosType end(1);

      Kokkos::resize(einsplines, esp.nBlocks);
      einsplines = Kokkos::View<spline_type*>("einsplines", esp.nBlocks);

      RandomGenerator<T> myrandom(11);
      //Array<T, 3> coef_data(nx+3, ny+3, nz+3);
      Kokkos::View<T***> coef_data("coef_data", nx + 3, ny + 3, nz + 3);

      for (int i = 0; i < esp.nBlocks; ++i)
      {
	spline_type* pspline_temp;
        myAllocator.createMultiBspline(pspline_temp, T(0), start, end, ng, PERIODIC, esp.nSplinesPerBlock);
	einsplines(i) = *pspline_temp;
        if (init_random)
        {
          for (int j = 0; j < esp.nSplinesPerBlock; ++j)
          {
            // Generate different coefficients for each orbital
            myrandom.generate_uniform(coef_data.data(), coef_data.extent(0));
            myAllocator.setCoefficientsForOneOrbital(j, coef_data, &einsplines(i));
          }
        }
      }
    }
    resize();
  }

  inline void evaluate_v_i(const QMCT::PosType& p)
  {
    ScopedTimer local_timer(timer);
    tmp_pos = p;
    compute_engine.copy_A44();
    esp.is_copy = true;
    if (esp.nSplines > esp.nSplinesSerialThreshold_V)
      Kokkos::parallel_for("EinsplineSPO::evaluate_v_parallel",
                           policy_v_parallel_t(esp.nBlocks, 1, 32),
                           *this);
    else
      Kokkos::parallel_for("EinsplineSPO::evaluate_v_serial", policy_v_serial_t(esp.nBlocks, 1, 32), *this);

    esp.is_copy = false;
    //   auto u = Lattice.toUnit_floor(p);
    //   for (int i = 0; i < nBlocks; ++i)
    //    compute_engine.evaluate_v(&einsplines(i), u[0], u[1], u[2], psi(i).data(), nSplinesPerBlock);
  }

  /** evaluate psi */
//   inline void evaluate_v_pfor(const QMCT::PosType& p)
//   {
//     auto u = esp.lattice.toUnit_floor(p);
//     //Why is this in kokkos Imp
// #pragma omp for nowait
//     for (int i = 0; i < esp.nBlocks; ++i)
//       compute_engine.evaluate_v(&einsplines(i), u[0], u[1], u[2], psi(i).data(), esp.nSplinesPerBlock);
//   }

  void evaluate_vgl_i(const QMCT::PosType& p)
  {
    //This didn't appear to be actually implemented for Kokkos
    // auto u = esp.lattice.toUnit_floor(p);
    //   for (int i = 0; i < esp.nBlocks; ++i)
    //     compute_engine.evaluate_vgl(&einsplines(i),
    //                               u[0],
    //                               u[1],
    //                               u[2],
    //                               psi(i).data(),
    //                               grad(i).data(),
    //                               hess(i).data(),
    // 				    esp.nSplinesPerBlock);
  }

  /** evaluate psi, grad and hess */
  void evaluate_vgh_i(const QMCT::PosType& p)
  {

    tmp_pos = p;

    esp.is_copy = true;
    compute_engine.copy_A44();

    if (esp.nSplines > esp.nSplinesSerialThreshold_VGH)
      Kokkos::parallel_for("EinsplineSPO::evalute_vgh", policy_vgh_parallel_t(esp.nBlocks, 1, 32), *this);
    else
      Kokkos::parallel_for("EinsplineSPO::evalute_vgh", policy_vgh_serial_t(esp.nBlocks, 1, 32), *this);
    esp.is_copy = false;
    //auto u = Lattice.toUnit_floor(p);
    //for (int i = 0; i < nBlocks; ++i)
    //  compute_engine.evaluate_vgh(&einsplines(i), u[0], u[1], u[2],
    //                              psi(i).data(), grad(i).data(), hess(i).data(),
    //                              nSplinesPerBlock);
  }

  inline void setLattice_i(const Tensor<T ,3>& lattice_b)
  {
    esp.lattice.set(lattice_b);
  }

  KOKKOS_INLINE_FUNCTION const EinsplineSPOParams<T>& getParams_i() const
  {
    return esp;
  }

  KOKKOS_INLINE_FUNCTION T getPsi_i(int ib, int n)
  {
    return psi[ib][n];
  }

  KOKKOS_INLINE_FUNCTION T getGrad_i(int ib, int n, int m)
  {
    return grad[ib](n,m);
  }

  KOKKOS_INLINE_FUNCTION T getHess_i(int ib, int n, int m)
  {
    return hess[ib](n,m);
  }

  KOKKOS_INLINE_FUNCTION void* getEinspline_i(int i) const
  {
    return &einsplines(i);
  }
  
  KOKKOS_INLINE_FUNCTION
  void operator()(const EvaluateVTag&, const team_v_serial_t& team) const
  {}
  KOKKOS_INLINE_FUNCTION
  void operator()(const EvaluateVTag&, const team_v_parallel_t& team) const
  {}
  KOKKOS_INLINE_FUNCTION
  void operator()(const EvaluateVGHTag&, const team_vgh_parallel_t& team) const
  {
    int block               = team.league_rank();
    auto u                  = esp.lattice.toUnit_floor(tmp_pos);
    einsplines(block).coefs = einsplines(block).coefs_view.data();
    compute_engine
      .evaluate_v(team, &einsplines(block), u[0], u[1], u[2], psi(block).data(), psi(block).extent(0));
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const EvaluateVGHTag&, const team_vgh_serial_t& team) const
  {
    int block               = team.league_rank();
    auto u                  = esp.lattice.toUnit_floor(tmp_pos);
    einsplines(block).coefs = einsplines(block).coefs_view.data();
    compute_engine
      .evaluate_v(team, &einsplines(block), u[0], u[1], u[2], psi(block).data(), psi(block).extent(0));
  }
};

  //  extern template class EinsplineSPODeviceImp<Devices::KOKKOS, float>;
  //extern template class EinsplineSPODeviceImp<Devices::KOKKOS, double>;
  
// template<Devices DT, typename T>
// KOKKOS_INLINE_FUNCTION void EinsplineSPODeviceImp<Devices::KOKKOS, T>::
// operator()(const typename EinsplineSPO<Devices::KOKKOS, T>::EvaluateVTag&,
// 	   const typename EinsplineSPODeviceImp<Devices::KOKKOS, T>::team_v_serial_t& team) const
// {
// }

// template<Devices DT, typename T>
// KOKKOS_INLINE_FUNCTION void EinsplineSPODeviceImp<Devices::KOKKOS, T>::
// operator()(const typename EinsplineSPODeviceImp<Devices::KOKKOS, T>::EvaluateVTag&,
// 	   const typename EinsplineSPODeviceImp<Devices::KOKKOS, T>::team_v_parallel_t& team) const
// {
// }
  
// template<typename T>
// KOKKOS_INLINE_FUNCTION void EinsplineSPODeviceImp<Devices::KOKKOS, T>::
// operator()(const typename EinsplineSPODeviceImp<Devices::KOKKOS, T>::EvaluateVTag&,
// 	   const typename EinsplineSPO<Devices::KOKKOS, T>::team_v_serial_t& team) const
// {
//   int block               = team.league_rank();
//   auto u                  = Lattice.toUnit_floor(tmp_pos);
//   einsplines(block).coefs = einsplines(block).coefs_view.data();
//   compute_engine
//       .evaluate_v(team, &einsplines(block), u[0], u[1], u[2], psi(block).data(), psi(block).extent(0));
// }

// template<typename T>
// KOKKOS_INLINE_FUNCTION void EinsplineSPODeviceImp<Devices::KOKKOS, T>::
// operator()(const typename EinsplineSPO<Devices::KOKKOS, T>::eEvaluateVTag&,
// 	   const typename EinsplineSPO<Devices::KOKKOS, T>::team_v_parallel_t& team) const
// {
//   int block               = team.league_rank();
//   auto u                  = Lattice.toUnit_floor(tmp_pos);
//   einsplines(block).coefs = einsplines(block).coefs_view.data();
//   compute_engine
//       .evaluate_v(team, &einsplines(block), u[0], u[1], u[2], psi(block).data(), psi(block).extent(0));
// }
  
} // namespace qmcpluplus

#endif
