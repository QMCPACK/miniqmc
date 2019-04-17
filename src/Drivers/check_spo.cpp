////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by:
// Jeongnim Kim, jeongnim.kim@intel.com,
//    Intel Corp.
// Amrita Mathuriya, amrita.mathuriya@intel.com,
//    Intel Corp.
//
// File created by:
// Jeongnim Kim, jeongnim.kim@intel.com,
//    Intel Corp.
////////////////////////////////////////////////////////////////////////////////
// -*- C++ -*-
/** @file check_spo.cpp
 * @brief Miniapp to check 3D spline implementation against the reference.
 */
#include <boost/hana/for_each.hpp>

#include <Utilities/Configuration.h>
#include <Utilities/Communicate.h>
#include <Particle/ParticleSet.h>
#include <Particle/ParticleSet_builder.hpp>
#include <Utilities/RandomGenerator.h>
#include <Input/Input.hpp>
#include <QMCWaveFunctions/EinsplineSPO.hpp>
#include "QMCWaveFunctions/EinsplineSPODevice.hpp"
#include "QMCWaveFunctions/EinsplineSPODeviceImp.hpp"
#include <QMCWaveFunctions/einspline_spo_ref.hpp>
#include <Utilities/qmcpack_version.h>
#include <getopt.h>
#include "Drivers/check_spo.h"
// #ifdef QMC_USE_CUDA
// #include "Drivers/test/CheckSPOStepsCUDA.hpp"
// #endif

using namespace std;

namespace qmcplusplus
{

void CheckSPOTest::printHelp()
{
  // clang-format off
  app_summary() << "usage:" << '\n';
  app_summary() << "  check_spo [-hvV] [-g \"n0 n1 n2\"] [-m meshfactor]" << '\n';
  app_summary() << "            [-n steps] [-r rmax] [-s seed]" << '\n';
  app_summary() << "options:" << '\n';
  app_summary() << "  -a  number of tiles                default: number of orbs\n";
  app_summary() << "  -d  device number                  default: 0 (CPU)\n";
  app_summary() << "  -g  set the 3D tiling.             default: 1 1 1" << '\n';
  app_summary() << "  -h  print help and exit" << '\n';
  app_summary() << "  -m  meshfactor                     default: 1.0" << '\n';
  app_summary() << "  -n  number of MC steps             default: 5" << '\n';
  app_summary() << "  -r  set the Rmax.                  default: 1.7" << '\n';
  app_summary() << "  -s  set the random seed.           default: 11" << '\n';
  app_summary() << "  -v  verbose output" << '\n';
  app_summary() << "  -V  print version information and exit" << '\n';
  //clang-format on

  //exit(1); // print help and exit
}

void CheckSPOTest::setup(int argc, char** argv)
{
      int opt;
    while (optind < argc)
    {
      if ((opt = getopt(argc, argv, "hvVa:c:d:f:g:m:n:r:s:")) != -1)
      {
        switch (opt)
        {
        case 'a':
          tileSize = atoi(optarg);
          break;
        case 'c': // number of members per team
          team_size = atoi(optarg);
          break;
	case 'd':
	  device = atoi(optarg);
	  break;
        case 'g': // tiling1 tiling2 tiling3
          sscanf(optarg, "%d %d %d", &na, &nb, &nc);
          break;
        case 'h':
          printHelp();
	  abort();
          break;
        case 'm':
        {
          const RealType meshfactor = atof(optarg);
          nx *= meshfactor;
          ny *= meshfactor;
          nz *= meshfactor;
        }
        break;
        case 'n':
          nsteps = atoi(optarg);
          break;
        case 'r': // rmax
          Rmax = atof(optarg);
          break;
        case 's':
          iseed = atoi(optarg);
          break;
        case 'v':
          verbose = true;
          break;
        case 'V':
          print_version(true);
	  exit(0);
          break;
        default:
          printHelp();
	  exit(0);
        }
      }
      else // disallow non-option arguments
      {
        app_error() << "Non-option arguments not allowed" << endl;
        printHelp();
	exit(EXIT_FAILURE);
      }
    }

    if (verbose)
      OutputManagerClass::get().setVerbosity(Verbosity::HIGH);
    else
      OutputManagerClass::get().setVerbosity(Verbosity::LOW);

  print_version(verbose);

    tmat = Tensor<int, 3>{na, 0, 0, 0, nb, 0, 0, 0, nc};
}

int CheckSPOTest::runTests()
{
  error = 0;
  if(device >= 0)
    {
      std::cout << "call handler\n";
      using Handler = decltype(hana::unpack(devices_range, hana::template_<CaseHandler>))::type;
      Handler handler(*this);
      handler.test(error,
									       team_size,
			     tmat,
			     tileSize,
			     nx,
			     ny,
			     nz,
			     nsteps,

		   Rmax, device);
    }
  else
    hana::for_each(devices_range,
		 [&](auto x) {
		 CheckSPOSteps<static_cast<Devices>(decltype(x)::value)>::test(error,
									       team_size,
			     tmat,
			     tileSize,
			     nx,
			     ny,
			     nz,
			     nsteps,

									       Rmax);});
  
  if(error > 0)
    return 1;
  else
    return 0;

}

// #ifdef QMC_USE_KOKKOS
// template class CheckSPOSteps<Devices::KOKKOS>;
// #endif

// #ifdef QMC_USE_CUDA
// template class CheckSPOSteps<Devices::CUDA>;
// #endif

}

using namespace qmcplusplus;

int main(int argc, char** argv)
{
  int error_code=0;
  // This is necessary because Kokkos needs to be initialized.
  hana::for_each(devices_range,
		 [&](auto x) {
		   CheckSPOSteps<static_cast<Devices>(decltype(x)::value)>::initialize(argc, argv);
		       });

  CheckSPOTest test;
  test.setup(argc, argv);
  error_code = test.runTests();
  hana::for_each(devices_range,
		 [&](auto x) {
		   CheckSPOSteps<static_cast<Devices>(decltype(x)::value)>::finalize();
		       });
  return error_code;
}

