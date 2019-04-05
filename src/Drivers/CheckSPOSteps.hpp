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
#include "check_spo.h"
#include "Utilities/Configuration.h"
#include "QMCWaveFunctions/einspline_spo_ref.hpp"
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
  static void initialize(int arc, char** argv) {};
  static void test(int& error,
                   int team_size,
                   const Tensor<int, 3>& tmat,
                   int tileSize,
                   const int nx,
                   const int ny,
                   const int nz,
                   const int nsteps,
                   const QMCT::RealType Rmax);
  static void finalize() {};

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
				const int tile_size,
                                const Tensor<OHMMS_PRECISION, 3>& lattice_b);

  template<typename T>
  static CheckSPOData<T> runThreads(const int team_size,
                                    ParticleSet& ions,
                                    const SPODevImp& spo_main,
                                    const SPORef& spo_ref_main,
                                    const int nsteps,
                                    const T Rmax);
};

extern template class CheckSPOSteps<Devices::CPU>;

} // namespace qmcplusplus


///////////////////////////////////////////////////
/** from here we have explicit instantiation declarations whose purpose is to speed compilation.
 */
#ifdef QMC_USE_KOKKOS
#include "Drivers/test/CheckSPOStepsKOKKOS.hpp"
#endif

#ifdef QMC_USE_CUDA
#include "Drivers/test/CheckSPOStepsCUDA.hpp"
namespace qmcplusplus
{
extern template void CheckSPOSteps<Devices::CUDA>::test(int&, int, qmcplusplus::Tensor<int, 3u> const&, int, int, int, int, int, double);
extern template typename CheckSPOSteps<Devices::CUDA>::SPODevImp
CheckSPOSteps<Devices::CUDA>::buildSPOMain(const int nx,
				const int ny,
				const int nz,
				const int norb,
				const int nTiles,
					   const int tile_size,
				const Tensor<OHMMS_PRECISION, 3>& lattice_b);

}
#endif


#endif
