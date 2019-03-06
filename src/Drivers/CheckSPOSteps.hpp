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

#ifndef QMCPLUSPLUS_CHECK_SPO_STEPS_HPP
#define QMCPLUSPLUS_CHECK_SPO_STEPS_HPP
#include "Devices.h"
#include "Utilities/Configuration.h"
#include "QMCWaveFunctions/EinsplineSPO.hpp"
#include "QMCWaveFunctions/EinsplineSPODevice.hpp"
#include "QMCWaveFunctions/EinsplineSPODeviceImp.hpp"
#include "Drivers/CheckSPOData.hpp"

namespace qmcplusplus
{
template<Devices DT>
class CheckSPOSteps
{
public:
  using SPODevImp = EinsplineSPO<DT, OHMMS_PRECISION>;
  using SPORef    = miniqmcreference::EinsplineSPO_ref<OHMMS_PRECISION>;

public:
  using QMCT = QMCTraits;
  static void initialize(int arc, char** argv);
  static void test(int& error,
                   int team_size,
                   const Tensor<int, 3>& tmat,
                   int tileSize,
                   const int nx,
                   const int ny,
                   const int nz,
                   const int nsteps,
                   const QMCT::RealType Rmax);
  static void finalize();

  template<typename T>
  static void thread_main(const int num_threads,
                          const int thread_id,
                          const int team_size,
                          const ParticleSet ions,
                          const SPODevImp spo_main,
                          const SPORef spo_ref_main,
                          const int nsteps,
                          const QMCT::RealType Rmax,
                          T& ratio,
                          T& nspheremoves,
                          T& dNumVGHCalls,
                          T& evalV_v_err,
                          T& evalVGH_v_err,
                          T& evalVGH_g_err,
                          T& evalVGH_h_err);

private:
  static SPODevImp buildSPOMain(const int nx,
                                const int ny,
                                const int nz,
                                const int norb,
                                const int nTiles,
                                const Tensor<OHMMS_PRECISION, 3>& lattice_b);

  template<typename T>
  static CheckSPOData<T> runThreads(const int team_size,
                                    ParticleSet& ions,
                                    const SPODevImp& spo_main,
                                    const SPORef& spo_ref_main,
                                    const int nsteps,
                                    const T Rmax);
};

} // namespace qmcplusplus
#endif
