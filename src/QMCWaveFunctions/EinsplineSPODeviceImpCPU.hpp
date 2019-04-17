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
 * @brief CPU implementation of EinsplineSPO
 */

#ifndef QMCPLUSPLUS_EINSPLINE_SPO_DEVICE_IMP_CPU_H
#define QMCPLUSPLUS_EINSPLINE_SPO_DEVICE_IMP_CPU_H

#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <vector>
#include <array>
#include <memory>
#include "Devices.h"
#include "clean_inlining.h"
#include "Numerics/Containers.h"
#include "Utilities/SIMD/allocator.hpp"
#include "Utilities/Configuration.h"
#include "Utilities/NewTimer.h"
#include "Utilities/RandomGenerator.h"
#include "QMCWaveFunctions/EinsplineSPODevice.hpp"
#include "QMCWaveFunctions/EinsplineSPODeviceImp.hpp"
#include "QMCWaveFunctions/EinsplineSPOParams.h"
#include "Numerics/Spline2/bspline_traits.hpp"
#include "Numerics/Spline2/BsplineAllocator.hpp"
#include "Numerics/Spline2/BsplineSet.hpp"
#include "Numerics/Spline2/MultiBsplineFuncs.hpp"
#include "Numerics/Spline2/MultiBsplineFuncs.hpp"

namespace qmcplusplus
{
template<typename T>
class EinsplineSPODeviceImp<Devices::CPU, T> : public EinsplineSPODevice<EinsplineSPODeviceImp<Devices::CPU, T>, T>
{
  using QMCT = QMCTraits;
  /// define the einspline data object type
  using spline_type     = typename bspline_traits<Devices::CPU, T, 3>::SplineType;
  using vContainer_type = aligned_vector<T>;
  using gContainer_type = VectorSoAContainer<T, 3>;
  using hContainer_type = VectorSoAContainer<T, 6>;
  using lattice_type    = CrystalLattice<T, 3>;

  /// use allocator
  //einspline::Allocator<Devices::CPU>& myAllocator;
  /// compute engine
  MultiBsplineFuncs<Devices::CPU, T> compute_engine;

  //using einspline_type = spline_type*;
  std::shared_ptr<BsplineSet<Devices::CPU, T>> einsplines;

  //  aligned_vector<vContainer_type> psi;
  aligned_vector<vContainer_type> psi;
  aligned_vector<gContainer_type> grad;
  aligned_vector<hContainer_type> hess;
  EinsplineSPOParams<T> esp;

public:
  EinsplineSPODeviceImp()
  {
    //std::cout << "EinsplineSPODeviceImpCPU() called" << '\n';
    esp.nBlocks    = 0;
    esp.nSplines   = 0;
    esp.firstBlock = 0;
    esp.lastBlock  = 0;
    einsplines     = nullptr;
    psi            = {};
    grad           = {};
    hess           = {};
  }

  /** CPU to CPU Constructor
   */
  EinsplineSPODeviceImp(const EinsplineSPODeviceImp<Devices::CPU, T>& in)
  {
    //std::cout << "EinsplineSPODeviceImpCPU Fat Copy constructor called" << '\n';
    const EinsplineSPOParams<T>& inesp = in.getParams();
    esp.nSplinesSerialThreshold_V      = inesp.nSplinesSerialThreshold_V;
    esp.nSplinesSerialThreshold_VGH    = inesp.nSplinesSerialThreshold_VGH;
    esp.nSplines                       = inesp.nSplines;
    esp.nSplinesPerBlock               = inesp.nSplinesPerBlock;
    esp.nBlocks                        = inesp.nBlocks;
    esp.firstBlock                     = 0;
    esp.lastBlock                      = inesp.nBlocks;
    esp.lattice                        = inesp.lattice;
    einsplines                         = in.einsplines;
    resize();
  }

  /** "Fat" Copy Constructor only supports CPU to CPU
   */
  EinsplineSPODeviceImp(const EinsplineSPODeviceImp<Devices::CPU, T>& in, int team_size, int member_id)
  {
    std::cout << "EinsplineSPODeviceImpCPU Fat Copy constructor called" << '\n';
    const EinsplineSPOParams<T>& inesp = in.getParams();
    esp.nSplinesSerialThreshold_V      = inesp.nSplinesSerialThreshold_V;
    esp.nSplinesSerialThreshold_VGH    = inesp.nSplinesSerialThreshold_VGH;
    esp.nSplines                       = inesp.nSplines;
    esp.nSplinesPerBlock               = inesp.nSplinesPerBlock;
    esp.nBlocks                        = (inesp.nBlocks + team_size - 1) / team_size;
    esp.firstBlock                     = esp.nBlocks * member_id;
    esp.lastBlock                      = std::min(inesp.nBlocks, esp.nBlocks * (member_id + 1));
    esp.nBlocks                        = esp.lastBlock - esp.firstBlock;
    esp.lattice                        = inesp.lattice;
    einsplines                         = in.einsplines;
    resize();
  }

  /// destructors
  ~EinsplineSPODeviceImp() {}

  /// resize the containers
  void resize()
  {
    if (esp.nBlocks > 0)
    {
      this->psi.resize(this->esp.nBlocks);
      this->grad.resize(esp.nBlocks);
      this->hess.resize(esp.nBlocks);
      for (int i = 0; i < esp.nBlocks; ++i)
      {
        this->psi[i].resize(esp.nSplinesPerBlock);
        this->grad[i].resize(esp.nSplinesPerBlock);
        this->hess[i].resize(esp.nSplinesPerBlock);
      }
    }
  }

  void set_i(int nx, int ny, int nz, int num_splines, int num_blocks, int splines_per_block, bool init_random = true)
  {
    this->esp.nSplines         = num_splines;
    this->esp.nBlocks          = num_blocks;
    this->esp.nSplinesPerBlock = splines_per_block;
    if (num_splines > splines_per_block * num_blocks)
      throw std::runtime_error("splines_per_block * nblocks < num_splines");

    this->esp.firstBlock = 0;
    this->esp.lastBlock  = esp.nBlocks;
    if (einsplines == nullptr)
    {
      QMCT::PosType start(0); // special constructor for 3d type
      QMCT::PosType end(1);   // special constructor for 3d type
      einsplines = std::make_shared<BsplineSet<Devices::CPU, T>>(esp.nBlocks);
      RandomGenerator<T> myrandom(11);
      Array<T, 3> coef_data(nx + 3, ny + 3, nz + 3);
      for (int i = 0; i < esp.nBlocks; ++i)
      {
        einsplines->creator()(i, start, end, nx, ny, nz, esp.nSplinesPerBlock);
        if (init_random)
        {
          for (int j = 0; j < esp.nSplinesPerBlock; ++j)
          {
            // Generate different coefficients for each orbital
            myrandom.generate_uniform(coef_data.data(), coef_data.size());
            einsplines->setCoefficientsForOneOrbital(j, coef_data, i);
          }
        }
      }
    }
    resize();
  }

  const EinsplineSPOParams<T>& getParams_i() const { return this->esp; }

  /** This is the proper way to hand back a shared pointer.  RVO insures this is efficient
   */
  std::shared_ptr<BsplineSet<Devices::CPU, T>> getEinsplines() const
  {
    return std::shared_ptr<BsplineSet<Devices::CPU, T>>(einsplines);
  }

  /** Consumer must makes sure the EinsplineSPO lives while you use this.
   */
  void* getEinspline_i(int i) const { return einsplines->get()[i]; }

  void setLattice_i(const Tensor<T, 3>& lattice) { esp.lattice.set(lattice); }

  inline void evaluate_v_i(const QMCT::PosType& p)
  {
    auto u                            = esp.lattice.toUnit_floor(p);
    std::vector<std::array<T, 3>> pos = {{u[0], u[1], u[2]}};
    for (int i = 0; i < esp.nBlocks; ++i)
      compute_engine.evaluate_v(einsplines->get()[i], pos, psi[i].data(), esp.nSplinesPerBlock);
  }

  inline void evaluate_vgh_i(const QMCT::PosType& p)
  {
    auto u                            = esp.lattice.toUnit_floor(p);
    std::vector<std::array<T, 3>> pos = {{u[0], u[1], u[2]}};
    for (int i = 0; i < esp.nBlocks; ++i)
    {
      compute_engine
          .evaluate_vgh(einsplines->get()[i], pos, psi[i].data(), grad[i].data(), hess[i].data(), esp.nSplinesPerBlock);
    }
  }

  void evaluate_vgl_i(const QMCT::PosType& p)
  {
    auto u                            = esp.lattice.toUnit_floor(p);
    std::vector<std::array<T, 3>> pos = {{u[0], u[1], u[2]}};
    for (int i = 0; i < esp.nBlocks; ++i)
      compute_engine
          .evaluate_vgl(einsplines->get()[i], pos, psi[i].data(), grad[i].data(), hess[i].data(), esp.nSplinesPerBlock);
  }

  T getPsi_i(int ib, int n) { return psi[ib][n]; }

  T getGrad_i(int ib, int n, int m) { return grad[ib].data(m)[n]; }

  T getHess_i(int ib, int n, int m) { return hess[ib].data(m)[n]; }
};

extern template class EinsplineSPODeviceImp<Devices::CPU, float>;
extern template class EinsplineSPODeviceImp<Devices::CPU, double>;

} // namespace qmcplusplus

#endif
