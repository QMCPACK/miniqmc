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

#ifndef QMCPLUSPLUS_CHECK_DETERMINANT_H
#define QMCPLUSPLUS_CHECK_DETERMINANT_H
#include "QMCWaveFunctions/Determinant.h"
#include "QMCWaveFunctions/DeterminantDevice.h"
#include "QMCWaveFunctions/DeterminantDeviceImp.h"
#include "Utilities/PrimeNumberSet.h"


namespace qmcplusplus
{

class CheckDeterminantTest
{
public:
  // clang-format off
  typedef QMCTraits::RealType           RealType;
  typedef ParticleSet::ParticlePos_t    ParticlePos_t;
  typedef ParticleSet::PosType          PosType;
  // clang-format on

  int na        = 1;
  int nb        = 1;
  int nc        = 1;
  int nsteps    = 5;
  int iseed     = 11;
  int nsubsteps = 1;
  int np        = 1;
  
  bool verbose = false;

  ParticleSet ions;
  Tensor<OHMMS_PRECISION, 3> lattice_b;
  int error;

  void setup(int argc, char** argv);

  int run_test();  
};

template<Devices DT>
class CheckDeterminantSteps
{
public:
  using QMCT = QMCTraits;
  static void initialize(int arc, char** argv);
  static double runThreads(int np, PrimeNumberSet<uint32_t>& myPrimes,
			  ParticleSet& ions, int& nsteps,
			 int& nsubsteps);
  static void test(int& error, ParticleSet& ions, int& nsteps, int& nsubsteps, int& np);
  static void finalize();
private:
  static void thread_main(const int ip,
			  const PrimeNumberSet<uint32_t>& myPrimes,
			  const ParticleSet& ions,
			  const int& nsteps,
			  const int& nsubsteps,
			  double& accumulated_error);
  static void updateFromDevice(DiracDeterminant<DeterminantDeviceImp<DT>>& determinant_device);
};
}
#ifdef QMC_USE_KOKKOS
#include "Drivers/test/CheckDeterminantHelpersKOKKOS.hpp"
  //extern template class CheckDeterminantHelpers<Devices::KOKKOS>;
#endif

#endif
