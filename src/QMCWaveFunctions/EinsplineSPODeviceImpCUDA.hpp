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
#include "QMCFutureTypes.hpp"
#include "clean_inlining.h"
#include "Numerics/Containers.h"
#include "Numerics/Spline2/BsplineSet.hpp"
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
#include "Numerics/Spline2/BsplineAllocator.hpp"
#include "Numerics/Spline2/MultiBsplineFuncs.hpp"
#include "Numerics/Spline2/MultiBsplineFuncsCUDA.hpp"

namespace qmcplusplus
{
/** Follows implementation in QMCPack in that EinsplineSPODeviceImp is fat CPUImp
 *  Except this one is of SoA CPU and I'm looking to trim or at least share generic code
 *  BDIM is batching dimension.
 */
template<typename T>
class EinsplineSPODeviceImp<Devices::CUDA, T> : public EinsplineSPODevice<EinsplineSPODeviceImp<Devices::CUDA, T>, T>
{
public:
  using QMCT     = QMCTraits;
  using ThisType = EinsplineSPODeviceImp<Devices::CUDA, T>;
  /// define the einspline data object type
  using device_spline_type = typename bspline_traits<Devices::CUDA, T, 3>::SplineType;
  using host_spline_type   = typename bspline_traits<Devices::CPU, T, 3>::SplineType;
  using vContainer_type    = aligned_vector<T>;
  using gContainer_type    = VectorSoAContainer<T, 3>;
  using lContainer_type    = VectorSoAContainer<T, 4>;
  using hContainer_type    = VectorSoAContainer<T, 6>;
  using lattice_type       = CrystalLattice<T, 3>;

  //using HessianParticipants = HessianParticipants<Devices::CUDA, T>;

  /// compute engine
  MultiBsplineFuncs<Devices::CUDA, T> compute_engine;

  std::shared_ptr<BsplineSet<Devices::CPU, T>> host_einsplines_;
  std::shared_ptr<BsplineSet<Devices::CUDA, T>> device_einsplines_;

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

  EinsplineSPOParams<T> esp;

  //device cache flags
  bool dirty_v;
  bool dirty_g;
  bool dirty_h;
  //PackContext pack_context;
public:
  EinsplineSPODeviceImp()
  {
    //std::cout << "EinsplineSPODeviceImpCPU() called" << '\n';
    esp.nBlocks    = 0;
    esp.nSplines   = 0;
    esp.firstBlock = 0;
    esp.lastBlock  = 0;
    esp.host_owner = false;
    esp.Owner      = false;
    esp.is_copy    = false;
    dirty_v        = false;
    dirty_g        = false;
    dirty_h        = false;
    psi            = {};
    grad           = {};
    hess           = {};
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
    dirty_v                            = false;
    dirty_g                            = false;
    dirty_h                            = false;
    device_einsplines_                 = std::make_shared<BsplineSet<Devices::CUDA, T>>(esp.nBlocks);
    host_einsplines_                   = in.getEinsplines();
    for (int i = 0; i < esp.nBlocks; ++i)
    {
      //std::cout << "EinsplineSPODeviceImp<Devices::CUDA> constructor, copy host_spline coefficients to device_spline coefficients\n";

      device_einsplines_->creator()(host_einsplines_->operator[](i), i);
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
    dirty_v                            = false;
    dirty_g                            = false;
    dirty_h                            = false;
    device_einsplines_                 = in.device_einsplines_;
    host_einsplines_                   = in.host_einsplines_;
    resize();
  }


  /** "Fat" Copy Constructor CPU to CUDA
   *  No support for team_size and member id.
   */
  EinsplineSPODeviceImp(const EinsplineSPODeviceImp<Devices::CPU, T>& in, int team_size, int member_id)
  {
    //std::cout << "EinsplineSPODeviceImpCUDA(EinsplineSPODeviceImp<Devices::CPU, T>&) Fat Copy constructor called"
    //       << '\n';
    const EinsplineSPOParams<T>& inesp = in.getParams();
    esp.nSplinesSerialThreshold_V      = inesp.nSplinesSerialThreshold_V;
    esp.nSplinesSerialThreshold_VGH    = inesp.nSplinesSerialThreshold_VGH;
    esp.nSplines                       = inesp.nSplines;
    esp.nSplinesPerBlock               = inesp.nSplinesPerBlock;
    esp.nBlocks                        = inesp.nBlocks; // (inesp.nBlocks + team_size - 1) / team_size;
    esp.firstBlock                     = 0;             //esp.nBlocks * member_id;
    esp.lastBlock                      = inesp.nBlocks; //std::min(inesp.nBlocks, esp.nBlocks * (member_id + 1));
    esp.lattice                        = inesp.lattice;
    dirty_v                            = false;
    dirty_g                            = false;
    dirty_h                            = false;

    device_einsplines_ = std::make_shared<BsplineSet<Devices::CUDA, T>>(esp.nBlocks);
    host_einsplines_   = in.getEinsplines();

    for (int i = 0, t = esp.firstBlock; i < esp.nBlocks; ++i, ++t)
    {
      device_einsplines_->creator()(host_einsplines_->operator[](i), i);
    }
    resize();
  }

  /** "Fat" Copy Constructor CUDA to CUDA
   */
  EinsplineSPODeviceImp(const EinsplineSPODeviceImp<Devices::CUDA, T>& in, int team_size, int member_id)
      : dev_psi(), dev_grad(), dev_linv(), dev_hess()
  {
    //std::cout << "EinsplineSPODeviceImpCUDA(EinsplineSPODeviceImp<Devices::CUDA, T>&,...) Fat Copy constructor called"
    //            << '\n';
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
    dirty_v                            = false;
    dirty_g                            = false;
    dirty_h                            = false;
    host_einsplines_                   = in.host_einsplines_;
    device_einsplines_                 = in.device_einsplines_;
    resize();
  }


  /// destructors
  ~EinsplineSPODeviceImp() {}

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
   *  calling this after anything but the default constructor
   *  Will drop this EinsplineSPOImp's connection to its SplineBundle, not reuse it
   */
  void set_i(int nx, int ny, int nz, int num_splines, int num_blocks, int splines_per_block, bool init_random = true)
  {
    this->esp.nSplines         = num_splines;
    this->esp.nBlocks          = num_blocks;
    this->esp.nSplinesPerBlock = splines_per_block;
    if (num_splines > splines_per_block * num_blocks)
      throw std::runtime_error("splines_per_block * nblocks < num_splines");
    // This would let you hold only some blocks
    // We never do this and I wouldn't do it this way.
    this->esp.firstBlock = 0;
    this->esp.lastBlock  = esp.nBlocks;

    std::cout << "Initializing CUDA Spline Coefficients with nBlocks: " << num_blocks
              << " and nSplinesPerblock : " << esp.nSplinesPerBlock << '\n';

    device_einsplines_ = std::make_shared<BsplineSet<Devices::CUDA, T>>(esp.nBlocks);
    host_einsplines_   = std::make_shared<BsplineSet<Devices::CPU, T>>(esp.nBlocks);

    TinyVector<int, 3> ng(nx, ny, nz);
    QMCT::PosType start(0);
    QMCT::PosType end(1);
    RandomGenerator<T> myrandom(11);
    Array<T, 3> coef_data(nx + 3, ny + 3, nz + 3);
    for (int i = 0; i < esp.nBlocks; ++i)
    {
      host_einsplines_->creator()(i, start, end, nx, ny, nz, esp.nSplinesPerBlock);
      if (init_random)
      {
        for (int j = 0; j < esp.nSplinesPerBlock; ++j)
        {
          // Generate different coefficients for each orbital
          myrandom.generate_uniform(coef_data.data(), coef_data.size());
          host_einsplines_->setCoefficientsForOneOrbital(j, coef_data, i);
        }
      }
    }

    for (int i = 0; i < esp.nBlocks; ++i)
    {
      device_einsplines_->creator()((*host_einsplines_)[i], i);
    }
    resize();
  }

  const EinsplineSPOParams<T>& getParams_i() const { return this->esp; }

  void* getEinspline_i(int i) const { return device_einsplines_->operator[](i); }
  void* getHostEinspline(int i) const { return host_einsplines_->operator[](i); }

  void setLattice_i(const Tensor<T, 3>& lattice) { esp.lattice.set(lattice); }

  /** Legacy single POS call
   */
  inline void evaluate_v_i(const QMCT::PosType& p)
  {
    dirty_v                           = true;
    auto u                            = esp.lattice.toUnit_floor(p);
    std::vector<std::array<T, 3>> pos = {{u[0], u[1], u[2]}};
    compute_engine.evaluate_v(device_einsplines_->operator[](0),
                              pos,
                              dev_psi.get_devptr(),
                              esp.nBlocks,
                              (size_t)esp.nSplinesPerBlock,
                              (size_t)esp.nSplinesPerBlock);
  }

  /** Legacy single POS call
   */
  inline void evaluate_vgh_i(const QMCT::PosType& p)
  {
    dirty_v                           = true;
    dirty_g                           = true;
    dirty_h                           = true;
    auto u                            = esp.lattice.toUnit_floor(p);
    std::vector<std::array<T, 3>> pos = {{u[0], u[1], u[2]}};
    cudaStream_t stream               = cudaStreamPerThread;
    compute_engine.evaluate_vgh(device_einsplines_->operator[](0),
                                pos,
                                dev_psi.get_devptr(),
                                dev_grad.get_devptr(),
                                dev_hess.get_devptr(),
                                esp.nBlocks,
                                esp.nSplines,
                                esp.nSplinesPerBlock,
                                stream);
  }

  inline HessianParticipants<Devices::CUDA, T> visit_for_vgh_i()
  {
    return HessianParticipants<Devices::CUDA, T>(*device_einsplines_, psi, grad, hess, dirty_v, dirty_g, dirty_h);
  }

  /** Legacy single POS call
   */
  void evaluate_vgl_i(const QMCT::PosType& p)
  {
    dirty_v                           = true;
    dirty_g                           = true;
    auto u                            = esp.lattice.toUnit_floor(p);
    std::vector<std::array<T, 3>> pos = {{u[0], u[1], u[2]}};

    compute_engine.evaluate_vgl(device_einsplines_->operator[](0),
                                pos,
                                dev_linv.get_devptr(),
                                dev_psi.get_devptr(),
                                dev_lapl.get_devptr(),
                                esp.nSplinesPerBlock);
  }

  T getPsi_i(int ib, int n)
  {
    if (dirty_v)
    {
      dev_psi.pull(psi);
      dirty_v = false;
    }
    return psi[ib][n];
  }

  T getGrad_i(int ib, int m, int n)
  {
    if (dirty_g)
    {
      dev_grad.pull(grad);
      dirty_g = false;
    }
    return grad[ib].data(m)[n];
  }

  T getHess_i(int ib, int m, int n)
  {
    if (dirty_h)
    {
      dev_hess.pull(hess);
      dirty_h = false;
    }
    return hess[ib].data(m)[n];
  }

  std::shared_ptr<BsplineSet<Devices::CPU, T>> getHostEinsplines() const
  {
    return std::shared_ptr<BsplineSet<Devices::CPU, T>>(host_einsplines_);
  }

  std::shared_ptr<BsplineSet<Devices::CUDA, T>> getDeviceEinsplines() const
  {
    return std::shared_ptr<BsplineSet<Devices::CUDA, T>>(device_einsplines_);
  }
};

extern template class EinsplineSPODeviceImp<Devices::CUDA, float>;
extern template class EinsplineSPODeviceImp<Devices::CUDA, double>;

} // namespace qmcplusplus

#endif
