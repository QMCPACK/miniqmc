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

#include <boost/hana/fwd/define_struct.hpp>
#include "Devices.h"
#include "clean_inlining.h"
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include "Numerics/Containers.h"
#include "Utilities/SIMD/allocator.hpp"
#include "Utilities/Configuration.h"
#include "Utilities/NewTimer.h"
#include "Utilities/RandomGenerator.h"
#include "QMCWaveFunctions/EinsplineSPODevice.hpp"
#include "QMCWaveFunctions/EinsplineSPODeviceImp.hpp"
#include "QMCWaveFunctions/EinsplineSPOParams.h"
#include "Numerics/Spline2/bspline_traits.hpp"
#include "Numerics/Spline2/bspline_allocator.hpp"
#include "Numerics/Spline2/MultiBspline.hpp"

namespace qmcplusplus
{
template<typename T>
class EinsplineSPODeviceImp<Devices::CPU, T>
    : public EinsplineSPODevice<EinsplineSPODeviceImp<Devices::CPU, T>, T>
{
  using QMCT = QMCTraits;
  /// define the einsplie data object type
  using spline_type     = typename bspline_traits<Devices::CPU, T, 3>::SplineType;
  using vContainer_type = std::vector<T>; // aligned_vector<T>;
  using gContainer_type = VectorSoAContainer<T, 3>;
  using hContainer_type = VectorSoAContainer<T, 6>;
  using lattice_type    = CrystalLattice<T, 3>;

  /// use allocator
  einspline::Allocator<Devices::CPU> myAllocator;
  /// compute engine
  MultiBspline<Devices::CPU, T> compute_engine;

  //using einspline_type = spline_type*;
  aligned_vector<spline_type*> einsplines;
  //  aligned_vector<vContainer_type> psi;
  aligned_vector<vContainer_type> psi;
  aligned_vector<gContainer_type> grad;
  aligned_vector<hContainer_type> hess;

  EinsplineSPOParams<T> esp;

public:
  EinsplineSPODeviceImp()
  {
    std::cout << "EinsplineSPODeviceImpCPU() called" << '\n';
    esp.nBlocks    = 0;
    esp.nSplines   = 0;
    esp.firstBlock = 0;
    esp.lastBlock  = 0;
    esp.Owner      = false;
    einsplines     = {};
    psi            = {};
    grad           = {};
    hess           = {};
  }

  void construct()
  {
    std::cout << "EinsplineSPODeviceImpCPU::construct() called" << '\n';
    esp.nBlocks    = 0;
    esp.nSplines   = 0;
    esp.firstBlock = 0;
    esp.lastBlock  = 0;
    esp.Owner      = false;
    einsplines     = {};
    //psi = {};
    //grad = {};
    //hess = {};
  }

  //Copy Constructor only supports CPU to CPU
  EinsplineSPODeviceImp(const EinsplineSPODevice<EinsplineSPODeviceImp<Devices::CPU, T>, T>& in,
                        int team_size,
                        int member_id)
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
    einsplines.resize(esp.nBlocks);
    for (int i = 0, t = esp.firstBlock; i < esp.nBlocks; ++i, ++t)
      einsplines[i] = static_cast<spline_type*>(in.getEinspline(t));
    resize();
  }

  /// destructors
  ~EinsplineSPODeviceImp()
  {
    if (esp.Owner)
      for (int i = 0; i < esp.nBlocks; ++i)
        myAllocator.destroy(einsplines[i]);
  }

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

  void set(int nx, int ny, int nz, int num_splines, int nblocks, bool init_random = true)
  {
    this->esp.nSplines         = num_splines;
    this->esp.nBlocks          = nblocks;
    this->esp.nSplinesPerBlock = num_splines / nblocks;
    this->esp.firstBlock       = 0;
    this->esp.lastBlock        = esp.nBlocks;
    if (einsplines.empty())
    {
      this->esp.Owner = true;
      TinyVector<int, 3> ng(nx, ny, nz);
      QMCT::PosType start(0);
      QMCT::PosType end(1);
      einsplines.resize(esp.nBlocks);
      RandomGenerator<T> myrandom(11);
      Array<T, 3> coef_data(nx + 3, ny + 3, nz + 3);
      for (int i = 0; i < esp.nBlocks; ++i)
      {
        this->myAllocator
            .createMultiBspline(einsplines[i], T(0), start, end, ng, PERIODIC, esp.nSplinesPerBlock);
        if (init_random)
        {
          for (int j = 0; j < esp.nSplinesPerBlock; ++j)
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

  const EinsplineSPOParams<T>& getParams() const { return this->esp; }

  void* getEinspline(int i) const { return einsplines[i]; }

  void setLattice(const Tensor<T, 3>& lattice) { esp.lattice.set(lattice); }

  inline void evaluate_v(const QMCT::PosType& p)
  {
    auto u = esp.lattice.toUnit_floor(p);
    for (int i = 0; i < esp.nBlocks; ++i)
      compute_engine.evaluate_v(einsplines[i], u[0], u[1], u[2], psi[i].data(), esp.nSplinesPerBlock);
  }

  inline void evaluate_vgh(const QMCT::PosType& p)
  {
    auto u = esp.lattice.toUnit_floor(p);
    for (int i = 0; i < esp.nBlocks; ++i)
      compute_engine.evaluate_vgh(einsplines[i],
                                  u[0],
                                  u[1],
                                  u[2],
                                  psi[i].data(),
                                  grad[i].data(),
                                  hess[i].data(),
                                  esp.nSplinesPerBlock);
  }

  void evaluate_vgl(const QMCT::PosType& p)
  {
    auto u = esp.lattice.toUnit_floor(p);
    for (int i = 0; i < esp.nBlocks; ++i)
      compute_engine.evaluate_vgl(einsplines[i],
                                  u[0],
                                  u[1],
                                  u[2],
                                  psi[i].data(),
                                  grad[i].data(),
                                  hess[i].data(),
                                  esp.nSplinesPerBlock);
  }

  T getPsi(int ib, int n) { return psi[ib][n]; }

  T getGrad(int ib, int n, int m) { return grad[ib].data(n)[m]; }

  T getHess(int ib, int n, int m) { return hess[ib].data(n)[m]; }
};

extern template class EinsplineSPODeviceImp<Devices::CPU, float>;
extern template class EinsplineSPODeviceImp<Devices::CPU, double>;

} // namespace qmcplusplus

#endif
