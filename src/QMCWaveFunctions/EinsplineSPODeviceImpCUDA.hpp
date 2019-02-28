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
#include "Numerics/Spline2/MultiBspline.hpp"
#include "Numerics/Spline2/MultiBsplineCUDA.hpp"

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
  using hContainer_type  = VectorSoAContainer<T, 6>;
  using lattice_type     = CrystalLattice<T, 3>;

  /// use allocator
  einspline::Allocator<Devices::CUDA> myAllocator;
  einspline::Allocator<Devices::CPU> my_host_allocator;
  /// compute engine
  MultiBspline<Devices::CUDA, T> compute_engine;

  //using einspline_type = spline_type*;
  aligned_vector<host_spline_type*> host_einsplines;
  aligned_vector<spline_type*> einsplines;
  //  aligned_vector<vContainer_type> psi;
  aligned_vector<vContainer_type> psi;
  aligned_vector<gContainer_type> grad;
  aligned_vector<hContainer_type> hess;

  //device pointers
  GPUArray<T,1> dev_psi;
  GPUArray<T,3> dev_grad;
  GPUArray<T,6> dev_hess;
  GPUArray<T,1> dev_linv;
  //device memory pitches
  
  EinsplineSPOParams<T> esp;

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
    host_einsplines = {};
    psi             = {};
    grad            = {};
    hess            = {};
  }

  /** CPU to CUDA Constructor
   */
  EinsplineSPODeviceImp(const EinsplineSPODevice<EinsplineSPODeviceImp<Devices::CPU, T>, T>& in) : dev_psi(), dev_grad(), dev_linv(), dev_hess()
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
    host_einsplines.resize(esp.nBlocks);
    for (int i = 0, t = esp.firstBlock; i < esp.nBlocks; ++i, ++t)
      host_einsplines[i] = static_cast<host_spline_type*>(in.getEinspline(t));
    resize();
  }

  /** CUDA to CUDA Constructor
   */
  EinsplineSPODeviceImp(const EinsplineSPODevice<EinsplineSPODeviceImp<Devices::CUDA, T>, T>& in) : dev_psi(), dev_grad(), dev_hess(), dev_linv()
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
    host_einsplines.resize(esp.nBlocks);
    for (int i = 0, t = esp.firstBlock; i < esp.nBlocks; ++i, ++t)
      host_einsplines[i] = static_cast<host_spline_type*>(in.getEinspline(t));
    resize();
  }


  /** "Fat" Copy Constructor CPU to CUDA
   */
  EinsplineSPODeviceImp(const EinsplineSPODevice<EinsplineSPODeviceImp<Devices::CPU, T>,T>& in,
                        int team_size,
                        int member_id) : dev_psi(), dev_grad(), dev_linv(), dev_hess()
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
    host_einsplines.resize(esp.nBlocks);
    for (int i = 0, t = esp.firstBlock; i < esp.nBlocks; ++i, ++t)
      host_einsplines[i] = static_cast<host_spline_type*>(in.getEinspline(t));
    resize();
  }

  /** "Fat" Copy Constructor CUDA to CUDA
   */
  EinsplineSPODeviceImp(const EinsplineSPODevice<EinsplineSPODeviceImp<Devices::CUDA, T>, T>& in,
                        int team_size,
                        int member_id) : dev_psi(), dev_grad(), dev_linv(), dev_hess()
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
    host_einsplines.resize(esp.nBlocks);
    for (int i = 0, t = esp.firstBlock; i < esp.nBlocks; ++i, ++t)
    {
      const ThisType& in_cast = static_cast<const ThisType&>(in);
      host_einsplines[i]      = static_cast<host_spline_type*>(in_cast.getHostEinspline(t));
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
      resizeCUDA();
    }
  }

  
  void resizeCUDA()
  {
    dev_psi.resize(esp.nBlocks, esp.nSplinesPerBlock);
    dev_grad.resize(esp.nBlocks, esp.nSplinesPerBlock);
    dev_hess.resize(esp.nBlocks, esp.nSplinesPerBlock);
    dev_linv.resize(esp.nBlocks, esp.nSplinesPerBlock);
  }
  
  /** TBI This is a duplication of the set_i CPU version
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
      RandomGenerator<T> myrandom(11);
      Array<T, 3> coef_data(nx + 3, ny + 3, nz + 3);
      for (int i = 0; i < esp.nBlocks; ++i)
      {
        this->my_host_allocator.createMultiBspline(host_einsplines[i], T(0), start, end, ng, PERIODIC, esp.nSplinesPerBlock);        if (init_random)
        {
          for (int j = 0; j < esp.nSplinesPerBlock; ++j)
          {
            // Generate different coefficients for each orbital
            myrandom.generate_uniform(coef_data.data(), coef_data.size());
            my_host_allocator.setCoefficientsForOneOrbital(j, coef_data, host_einsplines[i]);
          }
        }
	T dummyT, dummyDT;
	myAllocator.createMultiBspline_3d(host_einsplines[i],einsplines[i],dummyT, dummyDT);
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
    auto u = esp.lattice.toUnit_floor(p);
    for (int i = 0; i < esp.nBlocks; ++i)
      compute_engine.evaluate_v(einsplines[i], u[0], u[1], u[2], dev_psi[i], (size_t)esp.nSplinesPerBlock);
  }

  inline void evaluate_vgh_i(const QMCT::PosType& p)
  {
    auto u = esp.lattice.toUnit_floor(p);
    for (int i = 0; i < esp.nBlocks; ++i)
    {
      compute_engine.evaluate_vgh(einsplines[i],
                                  u[0],
                                  u[1],
                                  u[2],
                                  dev_psi[i],
                                  dev_grad[i],
                                  dev_hess[i],
                                  esp.nSplinesPerBlock);
    }
  }

  void evaluate_vgl_i(const QMCT::PosType& p)
  {
    auto u = esp.lattice.toUnit_floor(p);
    for (int i = 0; i < esp.nBlocks; ++i)
      compute_engine.evaluate_vgl(einsplines[i],
                                  u[0],
                                  u[1],
                                  u[2],
				  dev_linv[i],
                                  dev_psi[i],
                                  dev_hess[i],
                                  esp.nSplinesPerBlock);
  }

  T getPsi_i(int ib, int n) { return psi[ib][n]; }

  T getGrad_i(int ib, int n, int m) { return grad[ib].data(m)[n]; }

  T getHess_i(int ib, int n, int m) { return hess[ib].data(m)[n]; }
};

extern template class EinsplineSPODeviceImp<Devices::CUDA, float>;
extern template class EinsplineSPODeviceImp<Devices::CUDA, double>;

} // namespace qmcplusplus

#endif
