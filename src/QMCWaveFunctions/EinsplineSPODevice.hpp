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

#ifndef QMCPLUSPLUS_EINSPLINE_SPO_DEVICE_H
#define QMCPLUSPLUS_EINSPLINE_SPO_DEVICE_H
#include "boost/hana/core.hpp"
#include "Devices.h"
#include "QMCFutureTypes.hpp"
#include "Utilities/Configuration.h"
#include "QMCWaveFunctions/EinsplineSPOParams.h"
#include "Numerics/Spline2/BsplineSet.hpp"

/** @file
 * CRTP base class for Einspline SPO Devices
 */
namespace qmcplusplus
{
  namespace hana = boost::hana;


  template<Devices DT, typename T>
  struct HessianParticipants
  {
    using QPT = QMCFutureTypes<T>;
    HessianParticipants(BsplineSet<DT, T>& device_einsplines_,
			aligned_vector<typename QPT::vContainer_type>& psi_,
			aligned_vector<typename QPT::gContainer_type>& grad_,
			aligned_vector<typename QPT::hContainer_type>& hess_,
			bool& dirty_v_,
			bool& dirty_g_,
			bool& dirty_h_
			) : device_einsplines(device_einsplines_), psi(psi_), grad(grad_), hess(hess_), dirty_v(dirty_v_), dirty_g(dirty_g_), dirty_h(dirty_h_) {}
    HessianParticipants(const HessianParticipants& ) = default;
    BsplineSet<Devices::CUDA, T>& device_einsplines;

    aligned_vector<typename QPT::vContainer_type>& psi;
    aligned_vector<typename QPT::gContainer_type>& grad;
    aligned_vector<typename QPT::hContainer_type>& hess;

    bool& dirty_v;
    bool& dirty_g;
    bool& dirty_h;
  };



template<class DEVICEIMP, typename T>
class EinsplineSPODevice
{
public:
  using QMCT = QMCTraits;

protected:
  EinsplineSPODevice(const EinsplineSPODevice<DEVICEIMP, T>& espd, int team_size, int member_id)
  {
    //std::cout << "EinsplineSPODevice Fat Copy constructor called\n";
  }

  EinsplineSPODevice(const EinsplineSPODevice<DEVICEIMP, T>& espd)
  {
    //std::cout << "EinsplineSPODevice Copy constructor called\n";
  }

  EinsplineSPODevice()
  {
    //std::cout << "EinsplineDevice() called \n";
  };

public:
  void set(int nx, int ny, int nz, int num_splines, int nblocks, bool init_random = true)
  {
    impl().set_i(nx, ny, nz, num_splines, nblocks, init_random);
  }

  inline void evaluate_v(const QMCT::PosType& p) { impl().evaluate_v_i(p); }

  //HessianParticipants<hana::int_c<static_cast<int>(DEVICEIMP::ThisDevice)>, T> 
  inline auto visit_for_vgh() { return impl().visit_for_vgh_i(); }
  
  inline void evaluate_vgh(const QMCT::PosType& p) { impl().evaluate_vgh_i(p); }

  void evaluate_vgl(const QMCT::PosType& p) { impl().evaluate_vgl_i(p); }

  T getPsi(int ib, int n) { return impl().getPsi_i(ib, n); }

  T getGrad(int ib, int n, int m) { return impl().getGrad_i(ib, n, m); }

  T getHess(int ib, int n, int m) { return impl().getHess_i(ib, n, m); }

  const EinsplineSPOParams<T>& getParams() const { return implConst().getParams_i(); }

  void setLattice(const Tensor<T, 3>& lattice) { impl().setLattice_i(lattice); }

  void* getEinspline(int i) const { return implConst().getEinspline_i(i); }

private:
  DEVICEIMP& impl() { return *static_cast<DEVICEIMP*>(this); }
  const DEVICEIMP& implConst() const { return *static_cast<const DEVICEIMP*>(this); }
};

} // namespace qmcplusplus

#endif
