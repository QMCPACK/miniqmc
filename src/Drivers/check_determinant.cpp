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

#include <Utilities/Configuration.h>
#include <Particle/ParticleSet.h>
#include <Particle/ParticleSet_builder.hpp>
#include <Utilities/PrimeNumberSet.h>
#include <Utilities/RandomGenerator.h>
#include <Utilities/qmcpack_version.h>
#include <Input/Input.hpp>
#include <QMCWaveFunctions/Determinant.h>
#ifdef FUTURE_WAVEFUNCTIONS
#include "QMCWaveFunctions/future/Determinant.h"
#include "QMCWaveFunctions/future/DeterminantDevice.h"
#include "QMCWaveFunctions/future/DeterminantDeviceImp.h"
#include <boost/hana/for_each.hpp>
#include <boost/hana/functional/apply.hpp>
#include "Drivers/check_determinant.h"
#endif
#include <QMCWaveFunctions/DeterminantRef.h>
#include <getopt.h>

using namespace std;
using namespace qmcplusplus;


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

#ifdef FUTURE_WAVEFUNCTIONS
namespace hana = boost::hana;

template<future::DeterminantDeviceType DT>
void CheckDeterminantHelpers<DT>::initialize(int argc, char** argv)
{}
  
template<>
void CheckDeterminantHelpers<future::DDT::KOKKOS>::initialize(int argc, char** argv)
{
  Kokkos::initialize(argc, argv);
}

template<future::DeterminantDeviceType DT>
void CheckDeterminantHelpers<DT>::updateFromDevice(future::DiracDeterminant<future::DeterminantDeviceImp<DT>>& determinant_device)
{
}

#ifdef QMC_USE_OMPOL
template<future::DeterminantDeviceType::OMPOL>
void CheckDeterminantHelpers<DT>::updateFromDevice(future::DiracDeterminant<DT>& determinant_device)
{
  determinant_device.transfer_from_device();
}
#endif

template<future::DeterminantDeviceType DT>
double CheckDeterminantHelpers<DT>::runThreads(int np,
							      PrimeNumberSet<uint32_t>& myPrimes,
							      ParticleSet& ions,
							      int& nsteps,
							      int& nsubsteps)
{
  double accumulated_error = 0.0;
  CheckDeterminantHelpers<DT>
    ::thread_main(1, myPrimes, ions, nsteps, nsubsteps, accumulated_error);
  return accumulated_error;
}

template<>
double CheckDeterminantHelpers<future::DDT::KOKKOS>::runThreads(int np,
							      PrimeNumberSet<uint32_t>& myPrimes,
							      ParticleSet& ions,
							      int& nsteps,
							      int& nsubsteps)
{
  auto main_function = KOKKOS_LAMBDA (int thread_id, double& accumulated_error)
  {
    printf(" thread_id = %d\n",thread_id);
    CheckDeterminantHelpers<future::DDT::KOKKOS>
    ::thread_main(thread_id, myPrimes, ions, nsteps, nsubsteps, accumulated_error);
  };
  double accumulated_error = 0.0;
#if defined(KOKKOS_ENABLE_OPENMP) && !defined(KOKKOS_ENABLE_CUDA)
  int num_threads = Kokkos::OpenMP::thread_pool_size();
  int ncrews = 1;   
  int crewsize = std::max(1,num_threads/ncrews); 
  printf(" In partition master with %d threads, %d crews.  Crewsize = %d \n",num_threads,ncrews,crewsize);
  Kokkos::parallel_reduce(crewsize, main_function, accumulated_error);
  //Kokkos::OpenMP::partition_master(main_function,nmovers,crewsize);
#else
  main_function(0, accumulated_error );
#endif  
  return accumulated_error;
}

#ifdef QMC_USE_OMPOL
template<>
CheckDeterminantHelpers<future::DDT:OMPOL>::runThreads(int np,
						       PrimeNumberSet<uint32_t>& myPrimes,
			  ParticleSet& ions, int& nsteps,
						       int& nsubsteps)
{
  double accumulated_error = 0.0;
#pragma omp parallel reduction(+ : accumulated_error)
  {
    accumulated_error += this->thread_main(PrimeNumberSet<uint32_t>& myPrimes,
		      ParticleSet& ions, int& nsteps,
		      int& nsubsteps)

    // accumulate error
    for (int i = 0; i < determinant_ref.size(); i++)
    {
      accumulated_error += std::fabs(determinant_ref(i) - determinant(i));
    }
  } // end of omp parallel

  return accumulated_error;
}
#endif

template<future::DeterminantDeviceType DT>
void CheckDeterminantHelpers<DT>::thread_main(const int ip,
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
  std::cout << "Reference" << '\n';
  determinant_ref.checkMatrix();

  future::DiracDeterminant<future::DeterminantDeviceImp<DT>> determinant_device(nels, random_th);
  std::string enum_name = future::ddt_names[hana::int_c<static_cast<int>(DT)>];
  std::cout << enum_name << '\n';
  determinant_device.checkMatrix();

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
    determinant_device.recompute();
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
	determinant_device.ratio(els, iel);
	// Accept/reject the trial move
	if (ur[iel] > accept) // MC
	{
	  // Update position, and update temporary storage
	  els.acceptMove(iel);
	  determinant_ref.acceptMove(els, iel);
	  determinant_device.acceptMove(els, iel);
	  my_accepted++;
	}
	else
	{
	  els.rejectMove(iel);
	}
      } // iel
    }   // substeps

    els.donePbyP();
  }

  CheckDeterminantHelpers<DT>::updateFromDevice(determinant_device);
  
  // accumulate error
  for (int i = 0; i < determinant_ref.size(); i++)
  {
    accumulated_error += std::fabs(determinant_ref(i) - determinant_device(i));
  }
  
}

template<future::DeterminantDeviceType DT>
void CheckDeterminantHelpers<DT>::test(int& error, ParticleSet& ions,
				       int& nsteps,
				       int& nsubsteps,
				       int& np)
{
  PrimeNumberSet<uint32_t> myPrimes;

  double accumulated_error;
  accumulated_error = CheckDeterminantHelpers<DT>::runThreads(np, myPrimes,
					    ions, nsteps,
					    nsubsteps);
  
//    } // end of omp parallel

  constexpr double small_err = std::numeric_limits<double>::epsilon() * 6e8;

  cout << "total accumulated error of " << accumulated_error << " for " << np << " procs" << '\n';

  int error_code = 0;
  if (accumulated_error / np > small_err)
  {
    cout << "Checking failed with accumulated error: " << accumulated_error / np << " > "
	 << small_err << '\n';
    error_code = 1;
  }
  error += error_code;
}

template<future::DeterminantDeviceType DT>
void CheckDeterminantHelpers<DT>::finalize()
{}
  
template<>
void CheckDeterminantHelpers<future::DDT::KOKKOS>::finalize()
{
  Kokkos::finalize();
}


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
      outputManager.setVerbosity(Verbosity::HIGH);
    else
      outputManager.setVerbosity(Verbosity::LOW);
  }

int CheckDeterminantTest::run_test()
  {
    error = 0;
    // ddt_range has the index range of implementations at compile time.
    hana::for_each(future::ddt_range,
		 [&](auto x) {
		   CheckDeterminantHelpers<static_cast<future::DDT>(decltype(x)::value)>::test(error, ions, nsteps,
											       nsubsteps, np);});

    if(error > 0)
      return 1;
    else
      return 0;
  }
  


// template<typename DT>
// void initialize(DT dt, int arc, char** argv)
// {}

// template<>
// void initialize(future::DeterminantDeviceImp<future::DDT::KOKKOS> dt, int argc, char** argv)
// {
//   Kokkos::initialize(argc, argv);
// }



int main(int argc, char** argv)
{
  int error_code=0;
  // hana::for_each(ddts, [&](auto x) {
  // 			 initialize(x, argc, argc);
  // 		       });
  hana::for_each(future::ddt_range,
		 [&](auto x) {
		   CheckDeterminantHelpers<static_cast<future::DDT>(decltype(x)::value)>::initialize(argc, argv);
		       });

  CheckDeterminantTest test;
  test.setup(argc, argv);
  error_code = test.run_test();
  hana::for_each(future::ddt_range,
		 [&](auto x) {
		   CheckDeterminantHelpers<static_cast<future::DDT>(decltype(x)::value)>::finalize();
		       });
  return error_code;
}

#else

int main(int argc, char** argv)
{
  int error_code=0;
  Kokkos::initialize(argc, argv);
  { //Begin kokkos block.

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
    int np        = omp_get_max_threads();

    PrimeNumberSet<uint32_t> myPrimes;

    bool verbose = false;

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
          return 1;
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

    Tensor<int, 3> tmat(na, 0, 0, 0, nb, 0, 0, 0, nc);

    // setup ions
    ParticleSet ions;
    Tensor<OHMMS_PRECISION, 3> lattice_b;
    build_ions(ions, tmat, lattice_b);

    print_version(verbose);

    if (verbose)
      outputManager.setVerbosity(Verbosity::HIGH);
    else
      outputManager.setVerbosity(Verbosity::LOW);

    double accumulated_error = 0.0;
    double accumulated_error_cpu = 0.0;
    //#pragma omp parallel reduction(+ : accumulated_error)
//    {
      int ip = omp_get_thread_num();

      // create generator within the thread
      RandomGenerator<RealType> random_th(myPrimes[ip]);

      ParticleSet els;
      build_els(els, ions, random_th);
      els.update();

      const int nions = ions.getTotalNum();
      const int nels  = els.getTotalNum();
      const int nels3 = 3 * nels;

      miniqmcreference::DiracDeterminantRef determinant_ref(nels, random_th);
      determinant_ref.checkMatrix();
      DiracDeterminant determinant(nels, random_th);
      determinant.checkMatrix();
      // For VMC, tau is large and should result in an acceptance ratio of roughly
      // 50%
      // For DMC, tau is small and should result in an acceptance ratio of 99%
      const RealType tau = 2.0;

      ParticlePos_t delta(nels);

      RealType sqrttau = std::sqrt(tau);
      RealType accept  = 0.5;

      aligned_vector<RealType> ur(nels);
      random_th.generate_uniform(ur.data(), nels);

      els.update();

      int my_accepted = 0;
      for (int mc = 0; mc < nsteps; ++mc)
      {
        determinant_ref.recompute();
        determinant.recompute();
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
            determinant.ratio(els, iel);
            // Accept/reject the trial move
            if (ur[iel] > accept) // MC
            {
              // Update position, and update temporary storage
              els.acceptMove(iel);
              determinant_ref.acceptMove(els, iel);
              determinant.acceptMove(els, iel);
              my_accepted++;
            }
            else
            {
              els.rejectMove(iel);
            }
          } // iel
        }   // substeps

        els.donePbyP();
      }

      // accumulate error
      for (int i = 0; i < determinant_ref.size(); i++)
      {
        accumulated_error += std::fabs(determinant_ref(i) - determinant(i));
      }
//    } // end of omp parallel

    constexpr double small_err = std::numeric_limits<double>::epsilon() * 6e8;

    cout << "total accumulated error of " << accumulated_error << " for " << np << " procs" << '\n';

    if (accumulated_error / np > small_err)
    {
      cout << "Checking failed with accumulated error: " << accumulated_error / np << " > "
           << small_err << '\n';
      error_code=1;
    }
    else
      cout << "All checks passed for determinant" << '\n';
   
  } //end kokkos block
  Kokkos::finalize();
  return error_code;
}
#endif
