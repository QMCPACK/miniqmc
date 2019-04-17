////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by: M. Graham Lopez, Oak Ridge National Lab
//
// File created by: Jeongnim Kim, Intel
////////////////////////////////////////////////////////////////////////////////
// -*- C++ -*-
// clang-format off
/** @file check_determinant.cpp


  Compares against a reference implementation for correctness.

 */

// clang-format on

#include <iomanip>
#include <iostream>
#include <Utilities/Configuration.h>
#include <Particle/ParticleSet.h>
#include <Particle/ParticleSet_builder.hpp>
#include <Utilities/PrimeNumberSet.h>
#include <Utilities/RandomGenerator.h>
#include <Utilities/qmcpack_version.h>
#include <Input/Input.hpp>
#include <QMCWaveFunctions/Determinant.h>
#include "QMCWaveFunctions/DeterminantDevice.h"
#include "QMCWaveFunctions/DeterminantDeviceImp.h"
#include <boost/hana/for_each.hpp>
#include "Drivers/check_determinant.h"
#include <QMCWaveFunctions/DeterminantRef.h>
#include <getopt.h>
#include "Devices.h"
using namespace std;
namespace qmcplusplus
{
void print_help()
{
  // clang-format off
  cout << "usage:" << '\n';
  cout << "  check_determinant [-hvV] [-g \"n0 n1 n2\"] [-n steps]" << '\n';
  cout << "             [-N substeps] [-s seed]" << '\n';
  cout << "options:" << '\n';
  cout << "  -g  set the 3D tiling.             default: 1 1 1" << '\n';
  cout << "  -h  print help and exit" << '\n';
  cout << "  -n  number of MC steps             default: 5" << '\n';
  cout << "  -N  number of MC substeps          default: 1" << '\n';
  cout << "  -s  set the random seed.           default: 11" << '\n';
  cout << "  -v  verbose output" << '\n';
  cout << "  -V  print version information and exit" << '\n';
  //clang-format on

  exit(1); // print help and exit
}

namespace hana = boost::hana;
auto device_defined = hana::is_valid([](auto&& p) -> decltype((void)p.defined) {});

template<Devices DT>
void CheckDeterminantSteps<DT>::initialize(int argc, char** argv)
{}

template<Devices DT>
void CheckDeterminantSteps<DT>::updateFromDevice(DiracDeterminant<DeterminantDeviceImp<DT>>& determinant_device)
{
}

template<Devices DT>
double CheckDeterminantSteps<DT>::runThreads(int np,
					       PrimeNumberSet<uint32_t>& myPrimes,
					       ParticleSet& ions,
					       int& nsteps,
					       int& nsubsteps)
{
  double accumulated_error = 0.0;
#pragma omp parallel reduction(+:accumulated_error)
  {
  CheckDeterminantSteps<DT>
    ::thread_main(omp_get_thread_num(), myPrimes, ions, nsteps, nsubsteps, accumulated_error);
  }
  return accumulated_error;
}

template<Devices DT>
void CheckDeterminantSteps<DT>::thread_main(const int ip,
					      const PrimeNumberSet<uint32_t>& myPrimes,
					      const ParticleSet& ions,
					      const int& nsteps,
					      const int& nsubsteps,
					      double& accumulated_error)
{
  // create generator within the thread
  RandomGenerator<QMCT::RealType> random_th(myPrimes[ip]);

  ParticleSet els;
  build_els(els, ions, random_th);
  els.update();

  const int nions = ions.getTotalNum();
  const int nels  = els.getTotalNum();
  const int nels3 = 3 * nels;

  miniqmcreference::DiracDeterminantRef determinant_ref(nels, random_th);
  //std::cout << "Reference" << '\n';
  determinant_ref.checkMatrix();

  DiracDeterminant<DeterminantDeviceImp<DT>> determinant_devimp(nels, random_th);
  determinant_devimp.checkMatrix();

  // For VMC, tau is large and should result in an acceptance ratio of roughly
  // 50%
  // For DMC, tau is small and should result in an acceptance ratio of 99%
  const QMCT::RealType tau = 2.0;

  typedef ParticleSet::ParticlePos_t    ParticlePos_t;
  typedef ParticleSet::PosType          PosType;

  ParticlePos_t delta(nels);
  
  QMCT::RealType sqrttau = std::sqrt(tau);
  QMCT::RealType accept  = 0.5;

  aligned_vector<QMCT::RealType> ur(nels);
  random_th.generate_uniform(ur.data(), nels);

  els.update();

  //double accumulated_error = 0.0;
  int error_code = 0;
  int my_accepted = 0;
  for (int mc = 0; mc < nsteps; ++mc)
  {
    determinant_ref.recompute();
    determinant_devimp.recompute();
    for (int l = 0; l < nsubsteps; ++l) // drift-and-diffusion
    {
      random_th.generate_normal(&delta[0][0], nels3);
      for (int iel = 0; iel < nels; ++iel)
      {
	// Operate on electron with index iel
	els.setActive(iel);

	// Construct trial move
	PosType dr   = sqrttau * delta[iel];
	bool isValid = els.makeMoveAndCheck(iel, dr);

	if (!isValid)
	  continue;

	// Compute gradient at the trial position

	determinant_ref.ratio(els, iel);
	determinant_devimp.ratio(els, iel);
	// Accept/reject the trial move
	if (ur[iel] > accept) // MC
	{
	  // Update position, and update temporary storage
	  els.acceptMove(iel);
	  determinant_ref.acceptMove(els, iel);
	  determinant_devimp.acceptMove(els, iel);
	  my_accepted++;
	}
	else
	{
	  els.rejectMove(iel);
	}
      } // iel
    }   // substeps

    //std::cout << "Accepted: " << my_accepted << '\n';
    els.donePbyP();
  }

  CheckDeterminantSteps<DT>::updateFromDevice(determinant_devimp);
  
  // accumulate error
  for (int i = 0; i < determinant_ref.size(); i++)
  {
    accumulated_error += std::fabs(determinant_ref(i) - determinant_devimp(i));
  }
  //std::cout << "Error: " << accumulated_error << '\n';
}

template<Devices DT>
void CheckDeterminantSteps<DT>::test(int& error, ParticleSet& ions,
				       int& nsteps,
				       int& nsubsteps,
				       int& np)
{
  PrimeNumberSet<uint32_t> myPrimes;

  std::string enum_name = device_names[hana::int_c<static_cast<int>(DT)>];
  std::cout << "Testing Determinant Device Implementation: "<< enum_name << '\n';
  
  double accumulated_error;
  accumulated_error = CheckDeterminantSteps<DT>::runThreads(np, myPrimes,
					    ions, nsteps,
					    nsubsteps);

  constexpr double small_err = std::numeric_limits<double>::epsilon() * 6e8;

  cout << "total accumulated error of " << std::scientific << std::setw(18)
       << std::setprecision(14) << accumulated_error << " for " << np << " procs" << '\n';

  int error_code = 0;
  if (accumulated_error / np > small_err)
  {
    cout << "Checking failed with accumulated error: " << accumulated_error / np << " > "
	 << small_err << '\n';
    error_code = 1;
  }
  error += error_code;
}

template<Devices DT>
void CheckDeterminantSteps<DT>::finalize()
{}

void CheckDeterminantTest::setup(int argc, char** argv)
{
  np = omp_get_max_threads();
  Tensor<int, 3> tmat(na, 0, 0, 0, nb, 0, 0, 0, nc);
  
  int opt;
  while (optind < argc)
    {
      if ((opt = getopt(argc, argv, "hvVg:n:N:r:s:")) != -1)
      {
        switch (opt)
        {
        case 'g': // tiling1 tiling2 tiling3
          sscanf(optarg, "%d %d %d", &na, &nb, &nc);
          break;
        case 'h':
          print_help();
          break;
        case 'n':
          nsteps = atoi(optarg);
          break;
        case 'N':
          nsubsteps = atoi(optarg);
          break;
        case 's':
          iseed = atoi(optarg);
          break;
        case 'v':
          verbose = true;
          break;
        case 'V':
          print_version(true);
          
          break;
        default:
          print_help();
        }
      }
      else // disallow non-option arguments
      {
        cerr << "Non-option arguments not allowed" << endl;
        print_help();
      }
    }
    
    // setup ions
    build_ions(ions, tmat, lattice_b);

    print_version(verbose);
 
    if (verbose)
      OutputManagerClass::get().setVerbosity(Verbosity::HIGH);
    else
      OutputManagerClass::get().setVerbosity(Verbosity::LOW);
  }

int CheckDeterminantTest::run_test()
  {
    error = 0;
    // ddt_range has the index range of implementations at compile time.
    hana::for_each(devices_range,
		 [&](auto x) {
		   CheckDeterminantSteps<static_cast<Devices>(decltype(x)::value)>::test(error, ions, nsteps,
											       nsubsteps, np);});

    if(error > 0)
      return 1;
    else
      return 0;
  }
  

#ifdef QMC_USE_KOKKOS
template class CheckDeterminantSteps<Devices::KOKKOS>;
#endif
}
using namespace qmcplusplus;


int main(int argc, char** argv)
{
  int error_code=0;
  /**
   * Run initialize on each DeterminantDeviceImp
   * assumes initializes do not conflict with each other
   * at the moment just due to Kokkos
   */
  hana::for_each(devices_range,
		 [&](auto x) {
		   CheckDeterminantSteps<static_cast<Devices>(decltype(x)::value)>::initialize(argc, argv);
		       });

  CheckDeterminantTest test;
  test.setup(argc, argv);
  error_code = test.run_test();
  /**
   * Run finalize on each DeterminantDeviceImp
   * assumes finalizes do not conflict with each other
   * at the moment just due to Kokkos
   */
  hana::for_each(devices_range,
		 [&](auto x) {
		   CheckDeterminantSteps<static_cast<Devices>(decltype(x)::value)>::finalize();
		       });
  return error_code;
}

