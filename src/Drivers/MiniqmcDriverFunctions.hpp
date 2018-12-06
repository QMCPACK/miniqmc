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

#ifndef QMCPLUSPLUS_MINIQMC_DRIVER_FUNCTIONS_HPP
#define QMCPLUSPLUS_MINIQMC_DRIVER_FUNCTIONS_HPP

#include <functional>
#include <boost/hana/map.hpp>
#include "Devices.h"
#include "Drivers/MiniqmcOptions.hpp"
#include "Drivers/Mover.hpp"
#include "Particle/ParticleSet.h"
#include "Input/nio.hpp"
#include "QMCWaveFunctions/Determinant.h"
#include "QMCWaveFunctions/DeterminantDeviceImp.h"
#include "QMCWaveFunctions/SPOSet.h"
#include "QMCWaveFunctions/SPOSet_builder.h"
#include "Utilities/qmcpack_version.h"
#include "Utilities/PrimeNumberSet.h"

namespace qmcplusplus
{

/** A purely functional class implementing miniqmcdriver functions
 *  This is a cleaner alternative to defining functions on the spot
 *  and using ifdef's to isolate different device/model specific code.
 */


template<Devices DT>
class MiniqmcDriverFunctions
{
public:
  using QMCT = QMCTraits;
  static void initialize(int arc, char** argv);
  static void buildSPOSet(SPOSet*& spo_set,
			  MiniqmcOptions& mq_opt,
			  const int norb,
			  const int nTiles,
			  const Tensor<OHMMS_PRECISION, 3>& lattice_b);
  static void runThreads(MiniqmcOptions& mq_opt,
                         const PrimeNumberSet<uint32_t>& myPrimes,
                         ParticleSet& ions,
			 const SPOSet* spo_main);
  static void finalize();
private:
  static void mover_info();
  static void thread_main(const int ip,
			  const int team_size,
			  MiniqmcOptions& mq_opt,
                          const PrimeNumberSet<uint32_t>& myPrimes,
                          ParticleSet ions,
			  const SPOSet* spo_main);
  static void updateFromDevice(DiracDeterminant<DeterminantDeviceImp<DT>>& determinant_device);
};

template<Devices DT>
void MiniqmcDriverFunctions<DT>::buildSPOSet(SPOSet*& spo_set,
                                             MiniqmcOptions& mq_opt,
                                             const int norb,
                                             const int nTiles,
                                             const Tensor<OHMMS_PRECISION, 3>& lattice_b)
{
  spo_set =
      SPOSetBuilder<DT>::build(mq_opt.useRef, mq_opt.nx, mq_opt.ny, mq_opt.nz, norb, nTiles, lattice_b);
}

// template<Devices DT>
// void MiniqmcDriverFunctions<DT>::mover_info()
// {}

// template<>
// void MiniqmcDriverFunctions<Devices::KOKKOS>::mover_info()
// {
//   printf(" partition_id = %d\n", partition_id);
// }


// std::function<void(int, char**)> initializeCPU = std::bind(&MiniqmcDriverFunctions<Devices::CPU>::initialize, &cpu_functions, 1_, 2_);
// std::function<void(int, char**)> initializeKOKKOS = std::bind(&MiniqmcDriverFunctions<Devices::CPU>::initialize, &kokkos_functions, 1_, 2_);

// auto mdfi_map = hana::make_map(hana::make_pair(hana::int_c<static_cast<int>(Devices::CPU)>,
// 					       hana::type_c<initializeCPU>),
// 			      hana::make_pair(hana::int_c<static_cast<int>(Devices::KOKKOS)>,
// 					      hana::type_c<initializeKOKKOS>));


// static constexpr auto device_map =
//     hana::apply(hana::make_map,
// 		hana::make_pair(hana::int_c<static_cast<int>(Devices::CPU)>,
// 				hana::type_c<MiniqmcDriverFunctions<Devices::CPU>>),
// #ifdef QMC_USE_KOKKOS
// 		hana::make_pair(hana::int_c<static_cast<int>(Devices::KOKKOS)>,
//                                 hana::type_c<MiniqmcDriverFunctions<Devices::KOKKOS>>),
// #endif
// #ifdef QMC_USE_OMPOL
//                 hana::make_pair(hana::int_c<static_cast<int>(Devices::OMPOL)>,
//                                 hana::type_c<MiniqmcDriverFunctions<Devices::OMPOL>>),
// #endif
//                 hana::make_pair(hana::int_c<static_cast<int>(Devices::LAST)>,
//                                 hana::type_c<MiniqmcDriverFunctions<Devices::CPU>>));


} // namespace qmcplusplus

#endif
