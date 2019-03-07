////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2019 QMCPACK developers.
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
 * @brief CUDA implementation of EinsplineSPO
 */

#ifndef QMCPLUSPLUS_EINSPLINE_SPO_DEVICE_IMP_CUDA_H
#define QMCPLUSPLUS_EINSPLINE_SPO_DEVICE_IMP_CUDA_H

#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <vector>
#include <array>
#include "Devices.h"
#include "clean_inlining.h"
#include "Numerics/Containers.h"
#include "Utilities/SIMD/allocator.hpp"
#include "Utilities/Configuration.h"
#include "Utilities/NewTimer.h"
#include "Utilities/RandomGenerator.h"
#include "CUDA/GPUArray.h"
#include "Numerics/Spline2/BsplineAllocatorCUDA.hpp"
#include "QMCWaveFunctions/EinsplineSPO.hpp"
#include "QMCWaveFunctions/EinsplineSPODevice.hpp"
#include "QMCWaveFunctions/EinsplineSPODeviceImp.hpp"
#include "QMCWaveFunctions/EinsplineSPOParams.h"
#include "Numerics/Spline2/bspline_traits.hpp"
#include "Numerics/Spline2/bspline_allocator.hpp"
#include "Numerics/Spline2/MultiBsplineFuncs.hpp"
#include "Numerics/Spline2/MultiBsplineFuncsCUDA.hpp"

namespace qmcplusplus
{
/** Follows implementation in QMCPack in that EinsplineSPODeviceImp is fat CPUImp
 *  Except this one is of SoA CPU and I'm looking to trim or at least share generic code
 */
template<typename T>
class EinsplineSPODeviceImp<Devices::CUDA, T> : public EinsplineSPODevice<EinsplineSPODeviceImp<Devices::CUDA, T>, T>
{
  using QMCT     = QMCTraits;
  using ThisType = EinsplineSPODeviceImp<Devices::CUDA, T>;
  /// define the einspline data object type
  using host_spline_type = typename bspline_traits<Devices::CPU, T, 3>::SplineType;
  using spline_type      = typename bspline_traits<Devices::CUDA, T, 3>::SplineType;
  using vContainer_type  = aligned_vector<T>;
  using gContainer_type  = VectorSoAContainer<T, 3>;
  using lContainer_type  = VectorSoAContainer<T, 4>;
  using hContainer_type  = VectorSoAContainer<T, 6>;
  using lattice_type     = CrystalLattice<T, 3>;

  /// use allocator
  einspline::Allocator<Devices::CUDA> myAllocator;
  einspline::Allocator<Devices::CPU> my_host_allocator;
  /// compute engine
  MultiBsplineFuncs<Devices::CUDA, T> compute_engine;

  //using einspline_type = spline_type*;
  aligned_vector<host_spline_type*> host_einsplines;
  aligned_vector<spline_type*> einsplines;
  //  aligned_vector<vContainer_type> psi;
  aligned_vector<vContainer_type> psi;
  aligned_vector<gContainer_type> grad;
  aligned_vector<hContainer_type> hess;
  aligned_vector<lContainer_type> lapl;

  //device pointers
  GPUArray<T, 1, 1> dev_psi;
  GPUArray<T, 3, 1> dev_grad;
  GPUArray<T, 4, 1> dev_lapl;
  GPUArray<T, 6, 1> dev_hess;
  GPUArray<T, 1, 1> dev_linv;
  //device memory pitches

  EinsplineSPOParams<T> esp;
  bool dirty;

public:
  EinsplineSPODeviceImp()
  {
    //std::cout << "EinsplineSPODeviceImpCPU() called" << '\n';
    esp.nBlocks     = 0;
    esp.nSplines    = 0;
    esp.firstBlock  = 0;
    esp.lastBlock   = 0;
    esp.host_owner  = false;
    esp.Owner       = false;
    esp.is_copy     = false;
    dirty           = false;
    host_einsplines = {};
    psi             = {};
    grad            = {};
    hess            = {};
  }

  /** CPU to CUDA Constructor
   */
  EinsplineSPODeviceImp(const EinsplineSPODeviceImp<Devices::CPU, T>& in)
  {
    const EinsplineSPOParams<T>& inesp = in.getParams();
    esp.nSplinesSerialThreshold_V      = inesp.nSplinesSerialThreshold_V;
    esp.nSplinesSerialThreshold_VGH    = inesp.nSplinesSerialThreshold_VGH;
    esp.nSplines                       = inesp.nSplines;
    esp.nSplinesPerBlock               = inesp.nSplinesPerBlock;
    esp.nBlocks                        = inesp.nBlocks;
    esp.firstBlock                     = 0;
    esp.lastBlock                      = inesp.nBlocks;
    esp.lattice                        = inesp.lattice;
    esp.is_copy                        = true;
    esp.host_owner                     = false;
    dirty                              = false;
    host_einsplines.resize(esp.nBlocks);
    einsplines.resize(esp.nBlocks);
    for (int i = 0, t = esp.firstBlock; i < esp.nBlocks; ++i, ++t)
    {
      const ThisType& in_cast = static_cast<const ThisType&>(in);
      host_einsplines[i]      = static_cast<host_spline_type*>(in_cast.getHostEinspline(t));
      T dummyT, dummyDT;
      myAllocator.createMultiBspline(host_einsplines[i], einsplines[i], dummyT, dummyDT);
    }
    resize();
  }

  /** CUDA to CUDA Constructor
   */
  EinsplineSPODeviceImp(const EinsplineSPODeviceImp<Devices::CUDA, T>& in)
  {
    const EinsplineSPOParams<T>& inesp = in.getParams();
    esp.nSplinesSerialThreshold_V      = inesp.nSplinesSerialThreshold_V;
    esp.nSplinesSerialThreshold_VGH    = inesp.nSplinesSerialThreshold_VGH;
    esp.nSplines                       = inesp.nSplines;
    esp.nSplinesPerBlock               = inesp.nSplinesPerBlock;
    esp.nBlocks                        = inesp.nBlocks;
    esp.firstBlock                     = 0;
    esp.lastBlock                      = inesp.nBlocks;
    esp.lattice                        = inesp.lattice;
    esp.is_copy                        = true;
    esp.host_owner                     = false;
    dirty                              = false;
    host_einsplines.resize(esp.nBlocks);
    einsplines.resize(esp.nBlocks);
    for (int i = 0, t = esp.firstBlock; i < esp.nBlocks; ++i, ++t)
    {
      const ThisType& in_cast = static_cast<const ThisType&>(in);
      host_einsplines[i]      = static_cast<host_spline_type*>(in_cast.getHostEinspline(t));
      T dummyT, dummyDT;
      myAllocator.createMultiBspline(host_einsplines[i], einsplines[i], dummyT, dummyDT);
    }
    //Each stream needs their own copy of spo_main

    resize();
  }


  /** "Fat" Copy Constructor CPU to CUDA
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
    esp.is_copy                        = true;
    esp.host_owner                     = false;
    dirty                              = false;
    host_einsplines.resize(esp.nBlocks);
    einsplines.resize(esp.nBlocks);
    for (int i = 0, t = esp.firstBlock; i < esp.nBlocks; ++i, ++t)
    {
      const ThisType& in_cast = static_cast<const ThisType&>(in);
      host_einsplines[i]      = static_cast<host_spline_type*>(in_cast.getHostEinspline(t));
      T dummyT, dummyDT;
      myAllocator.createMultiBspline(host_einsplines[i], einsplines[i], dummyT, dummyDT);
    }
    resize();
  }

  /** "Fat" Copy Constructor CUDA to CUDA
   */
  EinsplineSPODeviceImp(const EinsplineSPODeviceImp<Devices::CUDA, T>& in, int team_size, int member_id)
      : dev_psi(), dev_grad(), dev_linv(), dev_hess()
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
    esp.is_copy                        = true;
    esp.host_owner                     = false;
    dirty                              = false;
    host_einsplines.resize(esp.nBlocks);
    einsplines.resize(esp.nBlocks);
    for (int i = 0, t = esp.firstBlock; i < esp.nBlocks; ++i, ++t)
    {
      const ThisType& in_cast = static_cast<const ThisType&>(in);
      host_einsplines[i]      = static_cast<host_spline_type*>(in_cast.getHostEinspline(t));
      T dummyT, dummyDT;
      myAllocator.createMultiBspline(host_einsplines[i], einsplines[i], dummyT, dummyDT);
    }
    resize();
  }


  /// destructors
  ~EinsplineSPODeviceImp()
  {
    if (esp.host_owner)
      for (int i = 0; i < esp.nBlocks; ++i)
        my_host_allocator.destroy(host_einsplines[i]);
  }

  /** resize both CPU and GPU containers
   *  resizing the GPU containers is destructive.
   */
  void resize()
  {
    if (esp.nBlocks > 0)
    {
      this->psi.resize(esp.nBlocks);
      this->grad.resize(esp.nBlocks);
      this->hess.resize(esp.nBlocks);
      this->lapl.resize(esp.nBlocks);
      for (int i = 0; i < esp.nBlocks; ++i)
      {
        this->psi[i].resize(esp.nSplinesPerBlock);
        this->grad[i].resize(esp.nSplinesPerBlock);
        this->lapl[i].resize(esp.nSplinesPerBlock);
        this->hess[i].resize(esp.nSplinesPerBlock);
      }
      resizeCUDA();
    }
  }

  void resizeCUDA()
  {
    dev_psi.resize(esp.nBlocks, esp.nSplinesPerBlock);
    dev_grad.resize(esp.nBlocks, esp.nSplinesPerBlock);
    dev_lapl.resize(esp.nBlocks, esp.nSplinesPerBlock);
    dev_hess.resize(esp.nBlocks, esp.nSplinesPerBlock);
    dev_linv.resize(esp.nBlocks, esp.nSplinesPerBlock);
  }

  /** Allocates splines and sets the initial coefficients. 
   *  calling this after anything but the default constructor is suspect
   *  And should probably throw
   */
  void set_i(int nx, int ny, int nz, int num_splines, int nblocks, bool init_random = true)
  {
    this->esp.nSplines         = num_splines;
    this->esp.nBlocks          = nblocks;
    this->esp.nSplinesPerBlock = num_splines / nblocks;
    this->esp.firstBlock       = 0;
    this->esp.lastBlock        = esp.nBlocks;
    if (host_einsplines.empty())
    {
      this->esp.Owner = true;
      TinyVector<int, 3> ng(nx, ny, nz);
      QMCT::PosType start(0);
      QMCT::PosType end(1);
      host_einsplines.resize(esp.nBlocks);
      einsplines.resize(esp.nBlocks);
      RandomGenerator<T> myrandom(11);
      Array<T, 3> coef_data(nx + 3, ny + 3, nz + 3);
      for (int i = 0; i < esp.nBlocks; ++i)
      {
        this->my_host_allocator
            .createMultiBspline(host_einsplines[i], T(0), start, end, ng, PERIODIC, esp.nSplinesPerBlock);
        if (init_random)
        {
          for (int j = 0; j < esp.nSplinesPerBlock; ++j)
          {
            // Generate different coefficients for each orbital
            myrandom.generate_uniform(coef_data.data(), coef_data.size());
            my_host_allocator.setCoefficientsForOneOrbital(j, coef_data, host_einsplines[i]);
          }
        }
      }
    }
    if (einsplines.empty())
    {
      for (int i = 0; i < esp.nBlocks; ++i)
      {
        T dummyT, dummyDT;
        myAllocator.createMultiBspline(host_einsplines[i], einsplines[i], dummyT, dummyDT);
      }
    }
    resize();
  }

  const EinsplineSPOParams<T>& getParams_i() const { return this->esp; }

  void* getEinspline_i(int i) const { return einsplines[i]; }
  void* getHostEinspline(int i) const { return host_einsplines[i]; }

  void setLattice_i(const Tensor<T, 3>& lattice) { esp.lattice.set(lattice); }

  inline void evaluate_v_i(const QMCT::PosType& p)
  {
    dirty                             = true;
    auto u                            = esp.lattice.toUnit_floor(p);
    std::vector<std::array<T, 3>> pos = {{u[0], u[1], u[2]}};
    compute_engine.evaluate_v(einsplines[0], pos, dev_psi.get_devptr(), (size_t)esp.nSplinesPerBlock);
  }

  inline void evaluate_vgh_i(const QMCT::PosType& p)
  {
    dirty                             = true;
    auto u                            = esp.lattice.toUnit_floor(p);
    std::vector<std::array<T, 3>> pos = {{u[0], u[1], u[2]}};
    compute_engine.evaluate_vgh(einsplines[0], pos, dev_psi.get_devptr(), dev_grad.get_devptr(), dev_hess.get_devptr(), esp.nSplinesPerBlock);
  }

  void evaluate_vgl_i(const QMCT::PosType& p)
  {
    dirty                             = true;
    auto u                            = esp.lattice.toUnit_floor(p);
    std::vector<std::array<T, 3>> pos = {{u[0], u[1], u[2]}};
    
    compute_engine.evaluate_vgl(einsplines[0], pos, dev_linv.get_devptr(), dev_psi.get_devptr(), dev_lapl.get_devptr(), esp.nSplinesPerBlock);
  }

  T getPsi_i(int ib, int n)
  {
    if (dirty)
    {
      dev_psi.pull(psi);
      dirty = false;
    }
    return psi[ib][n];
  }

  T getGrad_i(int ib, int n, int m)
  {
    if (dirty)
    {
      dev_grad.pull(grad);
      dirty = false;
    }
    return grad[ib].data(m)[n];
  }

  T getHess_i(int ib, int n, int m)
  {
    if (dirty)
    {
      dev_hess.pull(hess);
      dirty = false;
    }
    return hess[ib].data(m)[n];
  }
};

extern template class EinsplineSPODeviceImp<Devices::CUDA, float>;
extern template class EinsplineSPODeviceImp<Devices::CUDA, double>;

} // namespace qmcplusplus

#endif
