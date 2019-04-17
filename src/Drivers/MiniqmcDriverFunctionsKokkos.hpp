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
 * @brief declarations of Kokkos specializations
 */

#ifndef QMCPLUSPLUS_MINIQMCDRIVERFUNCTIONS_KOKKOS_HPP
#define QMCPLUSPLUS_MINIQMCDRIVERFUNCTIONS_KOKKOS_HPP

template<>
void MiniqmcDriverFunctions<Devices::KOKKOS>::initialize(int argc, char** argv);

template<>
void MiniqmcDriverFunctions<Devices::KOKKOS>::runThreads(MiniqmcOptions& mq_opt,
                                                         const PrimeNumberSet<uint32_t>& myPrimes,
                                                         ParticleSet& ions,
                                                         const SPOSet* spo_main);

template<>
void MiniqmcDriverFunctions<Devices::KOKKOS>::movers_runThreads(MiniqmcOptions& mq_opt,
                         const PrimeNumberSet<uint32_t>& myPrimes,
                         ParticleSet& ions,
				const SPOSet* spo_main);

template<>
void MiniqmcDriverFunctions<Devices::KOKKOS>::thread_main(const int ip,
                                                          const int team_size,
                                                          MiniqmcOptions& mq_opt,
                                                          const PrimeNumberSet<uint32_t>& myPrimes,
                                                          ParticleSet ions,
                                                          const SPOSet* spo_main);

extern template class qmcplusplus::MiniqmcDriverFunctions<Devices::KOKKOS>;

#endif
