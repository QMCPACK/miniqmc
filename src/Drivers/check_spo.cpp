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
#ifdef QMC_USE_CUDA
#include "Drivers/test/CheckSPOStepsCUDA.hpp"
#endif

using namespace std;

namespace qmcplusplus
{
template<Devices DT>
void CheckSPOSteps<DT>::finalize()
{}

template<Devices DT>
void CheckSPOSteps<DT>::initialize(int argc, char** argv)
{}


template<Devices DT>
typename CheckSPOSteps<DT>::SPODevImp CheckSPOSteps<DT>::buildSPOMain(const int nx,
							     const int ny,
							     const int nz,
							     const int norb,
							     const int nTiles,
							     const Tensor<OHMMS_PRECISION, 3>& lattice_b)
{
  SPODevImp spo_main;
  spo_main.set(nx, ny, nz, norb, nTiles);
  spo_main.setLattice(lattice_b);
  return spo_main;
}

template<Devices DT>
void CheckSPOSteps<DT>::test(int& error,
                             const int team_size,
                             const Tensor<int, 3>& tmat,
                             int tileSize,
                             const int nx,
                             const int ny,
                             const int nz,
                             const int nsteps,
                             const double Rmax)
{
  std::string enum_name = device_names[hana::int_c<static_cast<int>(DT)>];
  std::cout << "Testing Determinant Device Implementation: " << enum_name << '\n';

  ParticleSet ions;
  Tensor<OHMMS_PRECISION, 3> lattice_b;
  build_ions(ions, tmat, lattice_b);
  const int norb = count_electrons(ions, 1) / 2;
  tileSize       = (tileSize > 0) ? tileSize : norb;
  int nTiles     = norb / tileSize;

  const size_t SPO_coeff_size    = static_cast<size_t>(norb) * (nx + 3) * (ny + 3) * (nz + 3) * sizeof(QMCT::RealType);
  const double SPO_coeff_size_MB = SPO_coeff_size * 1.0 / 1024 / 1024;

  app_summary() << "Number of orbitals/splines = " << norb << endl
                << "Tile size = " << tileSize << endl
                << "Number of tiles = " << nTiles << endl
                << "Rmax = " << Rmax << endl;
  app_summary() << "Iterations = " << nsteps << endl;
  app_summary() << "OpenMP threads = " << omp_get_max_threads() << endl;

  app_summary() << "\nSPO coefficients size = " << SPO_coeff_size << " bytes (" << SPO_coeff_size_MB << " MB)" << endl;
  
  SPODevImp spo_main = buildSPOMain(nx, ny, nz, norb, nTiles, lattice_b);
  SPORef spo_ref_main;
  spo_ref_main.set(nx, ny, nz, norb, nTiles);
  spo_ref_main.Lattice.set(lattice_b);

  CheckSPOData<OHMMS_PRECISION> check_data =
      CheckSPOSteps::runThreads(team_size, ions, spo_main, spo_ref_main, nsteps, Rmax);

  OutputManagerClass::get().resume();

  check_data.evalV_v_err /= check_data.nspheremoves;
  check_data.evalVGH_v_err /= check_data.dNumVGHCalls;
  check_data.evalVGH_g_err /= check_data.dNumVGHCalls;
  check_data.evalVGH_h_err /= check_data.dNumVGHCalls;

  int np                           = omp_get_max_threads();
  constexpr QMCT::RealType small_v = std::numeric_limits<QMCT::RealType>::epsilon() * 1e4;
  constexpr QMCT::RealType small_g = std::numeric_limits<QMCT::RealType>::epsilon() * 3e6;
  constexpr QMCT::RealType small_h = std::numeric_limits<QMCT::RealType>::epsilon() * 6e8;
  int nfail                        = 0;
  app_log() << std::endl;
  if (check_data.evalV_v_err / np > small_v)
  {
    app_log() << "Fail in evaluate_v, V error =" << check_data.evalV_v_err / np << std::endl;
    nfail = 1;
  }
  if (check_data.evalVGH_v_err / np > small_v)
  {
    app_log() << "Fail in evaluate_vgh, V error =" << check_data.evalVGH_v_err / np << std::endl;
    nfail += 1;
  }
  if (check_data.evalVGH_g_err / np > small_g)
  {
    app_log() << "Fail in evaluate_vgh, G error =" << check_data.evalVGH_g_err / np << std::endl;
    nfail += 1;
  }
  if (check_data.evalVGH_h_err / np > small_h)
  {
    app_log() << "Fail in evaluate_vgh, H error =" << check_data.evalVGH_h_err / np << std::endl;
    nfail += 1;
  }

  if (nfail == 0)
    app_log() << "All checks passed for spo" << std::endl;
}


template<Devices DT>
template<typename T>
CheckSPOData<T> CheckSPOSteps<DT>::runThreads(int team_size,
                                              ParticleSet& ions,
                                              const SPODevImp& spo_main,
                                              const SPORef& spo_ref_main,
                                              int nsteps,
                                              T Rmax)
{
  T ratio         = 0.0;
  T nspheremoves  = 0;
  T dNumVGHCalls  = 0;
  T evalV_v_err   = 0.0;
  T evalVGH_v_err = 0.0;
  T evalVGH_g_err = 0.0;
  T evalVGH_h_err = 0.0;

#pragma omp parallel reduction(+:ratio,nspheremoves,dNumVGHCalls) \
   reduction(+:evalV_v_err,evalVGH_v_err,evalVGH_g_err,evalVGH_h_err)
  {
    const int np = omp_get_num_threads();
    const int ip = omp_get_thread_num();
    CheckSPOSteps<DT>::thread_main(np,
                                   ip,
                                   team_size,
                                   ions,
                                   spo_main,
                                   spo_ref_main,
                                   nsteps,
                                   Rmax,
                                   ratio,
                                   nspheremoves,
                                   dNumVGHCalls,
                                   evalV_v_err,
                                   evalVGH_v_err,
                                   evalVGH_g_err,
                                   evalVGH_h_err);
  }
  return CheckSPOData<T>{ratio, nspheremoves, dNumVGHCalls, evalV_v_err, evalVGH_v_err, evalVGH_g_err, evalVGH_h_err};
}


/** The inside of the parallel reduction for test.
 *  Since kokkos could magically marshall over MPI
 *  we have to pass by copy even if that is inefficient
 *  for on node parallelism like here.
 */
template<Devices DT>
template<typename T>
void CheckSPOSteps<DT>::thread_main(const int np,
                                    const int ip,
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
                                    T& evalVGH_h_err)
{
  const int team_id   = ip / team_size;
  const int member_id = ip % team_size;

  // create generator within the thread
  RandomGenerator<QMCT::RealType> random_th(MakeSeed(team_id, np));

  ParticleSet els;
  build_els(els, ions, random_th);
  els.update();

  const int nions = ions.getTotalNum();
  const int nels  = els.getTotalNum();
  const int nels3 = 3 * nels;

  // create pseudopp
  NonLocalPP<OHMMS_PRECISION> ecp(random_th);
  // create spo per thread
  SPODevImp spo(spo_main, team_size,member_id);
  //SPODevImp& spo = *dynamic_cast<SPODevImp*>(SPOSetBuilder<DT>::buildView(false, spo_main, team_size, member_id));
  SPORef spo_ref(spo_ref_main, team_size, member_id);

  // use teams
  // if(team_size>1 && team_size>=nTiles ) spo.set_range(team_size,ip%team_size);

  // this is the cutoff from the non-local PP
  const int nknots(ecp.size());

  ParticleSet::ParticlePos_t delta(nels);
  ParticleSet::ParticlePos_t rOnSphere(nknots);

  QMCT::RealType sqrttau = 2.0;
  QMCT::RealType accept  = 0.5;

  vector<QMCT::RealType> ur(nels);
  random_th.generate_uniform(ur.data(), nels);
  const double zval = 1.0 * static_cast<double>(nels) / static_cast<double>(nions);

  int my_accepted = 0, my_vals = 0;

  EinsplineSPOParams<T> esp = spo.getParams();
  for (int mc = 0; mc < nsteps; ++mc)
  {
    random_th.generate_normal(&delta[0][0], nels3);
    random_th.generate_uniform(ur.data(), nels);
    for (int ib = 0; ib < esp.nBlocks; ib++)

      // VMC
      for (int iel = 0; iel < nels; ++iel)
      {
        QMCT::PosType pos = els.R[iel] + sqrttau * delta[iel];

        spo.evaluate_vgh(pos);
        spo_ref.evaluate_vgh(pos);
        // accumulate error
        for (int ib = 0; ib < esp.nBlocks; ib++)
          for (int n = 0; n < esp.nSplinesPerBlock; n++)
          {
            // value
            evalVGH_v_err += std::fabs(spo.getPsi(ib, n) - spo_ref.psi[ib][n]);
            // grad
            evalVGH_g_err += std::fabs(spo.getGrad(ib, n, 0) - spo_ref.grad[ib].data(0)[n]);
            evalVGH_g_err += std::fabs(spo.getGrad(ib, n, 1) - spo_ref.grad[ib].data(1)[n]);
            evalVGH_g_err += std::fabs(spo.getGrad(ib, n, 2) - spo_ref.grad[ib].data(2)[n]);
            // hess
            evalVGH_h_err += std::fabs(spo.getHess(ib, n, 0) - spo_ref.hess[ib].data(0)[n]);
            evalVGH_h_err += std::fabs(spo.getHess(ib, n, 1) - spo_ref.hess[ib].data(1)[n]);
            evalVGH_h_err += std::fabs(spo.getHess(ib, n, 2) - spo_ref.hess[ib].data(2)[n]);
            evalVGH_h_err += std::fabs(spo.getHess(ib, n, 3) - spo_ref.hess[ib].data(3)[n]);
            evalVGH_h_err += std::fabs(spo.getHess(ib, n, 4) - spo_ref.hess[ib].data(4)[n]);
            evalVGH_h_err += std::fabs(spo.getHess(ib, n, 5) - spo_ref.hess[ib].data(5)[n]);
          }
        if (ur[iel] < accept)
        {
          els.R[iel] = pos;
          my_accepted++;
        }
      }

    random_th.generate_uniform(ur.data(), nels);
    ecp.randomize(rOnSphere); // pick random sphere
    for (int iat = 0, kat = 0; iat < nions; ++iat)
    {
      const int nnF    = static_cast<int>(ur[kat++] * zval);
      QMCT::RealType r = Rmax * ur[kat++];
      auto centerP     = ions.R[iat];
      my_vals += (nnF * nknots);

      for (int nn = 0; nn < nnF; ++nn)
      {
        for (int k = 0; k < nknots; k++)
        {
          QMCT::PosType pos = centerP + r * rOnSphere[k];
          spo.evaluate_v(pos);
          spo_ref.evaluate_v(pos);
          // accumulate error
          for (int ib = 0; ib < esp.nBlocks; ib++)
            for (int n = 0; n < esp.nSplinesPerBlock; n++)
              evalV_v_err += std::fabs(spo.getPsi(ib, n) - spo_ref.psi[ib][n]);
        }
      } // els
    }   // ions

  } // steps.

  ratio += QMCT::RealType(my_accepted) / QMCT::RealType(nels * nsteps);
  nspheremoves += QMCT::RealType(my_vals) / QMCT::RealType(nsteps);
  dNumVGHCalls += nels;
}

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

#ifdef QMC_USE_KOKKOS
template class CheckSPOSteps<Devices::KOKKOS>;
#endif

#ifdef QMC_USE_CUDA
template class CheckSPOSteps<Devices::CUDA>;
#endif

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

