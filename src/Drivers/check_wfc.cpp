////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by:
// Jeongnim Kim, jeongnim.kim@intel.com,
//    Intel Corp.
// Ye Luo, yeluo@anl.gov,
//    Argonne National Laboratory
//
// File created by:
// Ye Luo, yeluo@anl.gov,
//    Argonne National Laboratory
////////////////////////////////////////////////////////////////////////////////
// -*- C++ -*-
/** @file check_wfc.cpp
 * @brief Miniapp to check individual wave function component against its
 * reference.
 */

#include <Utilities/Configuration.h>
#include <Particle/ParticleSet.h>
#include <Particle/DistanceTable.h>
#include <Numerics/Containers.h>
#include <Utilities/PrimeNumberSet.h>
#include <Utilities/RandomGenerator.h>
#include <Input/Input.hpp>
#include <QMCWaveFunctions/Jastrow/PolynomialFunctor3D.h>
#include <QMCWaveFunctions/Jastrow/ThreeBodyJastrowRef.h>
#include <QMCWaveFunctions/Jastrow/ThreeBodyJastrow.h>
#include <QMCWaveFunctions/Jastrow/BsplineFunctor.h>
#include <QMCWaveFunctions/Jastrow/OneBodyJastrowRef.h>
#include <QMCWaveFunctions/Jastrow/OneBodyJastrow.h>
#include <QMCWaveFunctions/Jastrow/TwoBodyJastrowRef.h>
#include <QMCWaveFunctions/Jastrow/TwoBodyJastrow.h>
#include <Utilities/qmcpack_version.h>
#include <getopt.h>

using namespace std;
using namespace qmcplusplus;

void print_help()
{
  //clang-format off
  cout << "usage:" << '\n';
  cout << "  check_wfc [-hvV] [-f wfc_component] [-g \"n0 n1 n2\"]"     << '\n';
  cout << "            [-r rmax] [-s seed]"                             << '\n';
  cout << "options:"                                                    << '\n';
  cout << "  -f  specify wavefunction component to check"               << '\n';
  cout << "      one of: J1, J2, J3.            default: J2"            << '\n';
  cout << "  -g  set the 3D tiling.             default: 1 1 1"         << '\n';
  cout << "  -h  print help and exit"                                   << '\n';
  cout << "  -r  set the Rmax.                  default: 1.7"           << '\n';
  cout << "  -s  set the random seed.           default: 11"            << '\n';
  cout << "  -v  verbose output"                                        << '\n';
  cout << "  -V  print version information and exit"                    << '\n';
  //clang-format on

  exit(1); // print help and exit
}

int main(int argc, char **argv)
{


  // clang-format off
  typedef QMCTraits::RealType           RealType;
  typedef ParticleSet::ParticlePos_t    ParticlePos_t;
  typedef ParticleSet::PosType          PosType;
  // clang-format on

  // use the global generator

  int na     = 1;
  int nb     = 1;
  int nc     = 1;
  int iseed   = 11;
  RealType Rmax(1.7);
  string wfc_name("J2");

  bool verbose = false;

  int opt;
  while(optind < argc)
  {
    if ((opt = getopt(argc, argv, "hvVf:g:r:s:")) != -1)
    {
      switch (opt)
      {
      case 'f': // Wave function component
        wfc_name = optarg;
        break;
      case 'g': // tiling1 tiling2 tiling3
        sscanf(optarg, "%d %d %d", &na, &nb, &nc);
        break;
      case 'h': print_help(); break;
      case 'r': // rmax
        Rmax = atof(optarg);
        break;
      case 's':
        iseed = atoi(optarg);
        break;
      case 'v': verbose = true; break;
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

  print_version(verbose);

  if (verbose) {
    outputManager.setVerbosity(Verbosity::HIGH);
  }

  if (wfc_name != "J1" && wfc_name != "J2" && wfc_name != "J3" &&
      wfc_name != "JeeI")
  {
    cerr << "Uknown wave funciton component:  " << wfc_name << endl << endl;
    print_help();
  }

  Random.init(0, 1, iseed);
  Tensor<int, 3> tmat(na, 0, 0, 0, nb, 0, 0, 0, nc);

  // turn off output
  if (omp_get_max_threads() > 1)
  {
    outputManager.shutOff();
  }

  // list of accumulated errors
  double evaluateLog_v_err = 0.0;
  double evaluateLog_g_err = 0.0;
  double evaluateLog_l_err = 0.0;
  double evalGrad_g_err    = 0.0;
  double ratioGrad_r_err   = 0.0;
  double ratioGrad_g_err   = 0.0;
  double evaluateGL_g_err  = 0.0;
  double evaluateGL_l_err  = 0.0;
  double ratio_err         = 0.0;

  PrimeNumberSet<uint32_t> myPrimes;

  // clang-format off
  #pragma omp parallel reduction(+:evaluateLog_v_err,evaluateLog_g_err,evaluateLog_l_err,evalGrad_g_err) \
   reduction(+:ratioGrad_r_err,ratioGrad_g_err,evaluateGL_g_err,evaluateGL_l_err,ratio_err)
  // clang-format on
  {
    ParticleSet ions, els;
    ions.setName("ion");
    els.setName("e");
    OHMMS_PRECISION scale = 1.0;

    int ip = omp_get_thread_num();

    // create generator within the thread
    Random.init(0, 1, iseed);
    RandomGenerator<RealType> random_th(myPrimes[ip]);

    tile_cell(ions, tmat, scale);
    ions.RSoA = ions.R; // fill the SoA

    const int nions = ions.getTotalNum();
    const int nels  = count_electrons(ions, 1);
    const int nels3 = 3 * nels;

    { // create up/down electrons
      els.Lattice.BoxBConds = 1;
      els.Lattice.set(ions.Lattice);
      vector<int> ud(2);
      ud[0] = nels / 2;
      ud[1] = nels - ud[0];
      els.create(ud);
      els.R.InUnit = 1;
      random_th.generate_uniform(&els.R[0][0], nels3);
      els.convert2Cart(els.R); // convert to Cartiesian
      els.RSoA = els.R;
    }

    ParticleSet els_ref(els);
    els_ref.RSoA = els_ref.R;

    // create tables
    DistanceTable::add(els, DT_SOA);
    DistanceTable::add(els_ref, DT_SOA);
    DistanceTableData *d_ie = DistanceTable::add(ions, els_ref, DT_SOA);
    d_ie->setRmax(Rmax);

    ParticlePos_t delta(nels);

    RealType sqrttau = 2.0;

    vector<RealType> ur(nels);
    random_th.generate_uniform(ur.data(), nels);

    WaveFunctionComponentBasePtr wfc     = nullptr;
    WaveFunctionComponentBasePtr wfc_ref = nullptr;
    if (wfc_name == "J2")
    {
      TwoBodyJastrow<BsplineFunctor<RealType>> *J =
          new TwoBodyJastrow<BsplineFunctor<RealType>>(els);
      buildJ2(*J, els.Lattice.WignerSeitzRadius);
      wfc = dynamic_cast<WaveFunctionComponentBasePtr>(J);
      cout << "Built J2" << endl;
      miniqmcreference::TwoBodyJastrowRef<BsplineFunctor<RealType>> *J_ref =
          new miniqmcreference::TwoBodyJastrowRef<BsplineFunctor<RealType>>(
              els_ref);
      buildJ2(*J_ref, els.Lattice.WignerSeitzRadius);
      wfc_ref = dynamic_cast<WaveFunctionComponentBasePtr>(J_ref);
      cout << "Built J2_ref" << endl;
    }
    else if (wfc_name == "J1")
    {
      OneBodyJastrow<BsplineFunctor<RealType>> *J =
          new OneBodyJastrow<BsplineFunctor<RealType>>(ions, els);
      buildJ1(*J, els.Lattice.WignerSeitzRadius);
      wfc = dynamic_cast<WaveFunctionComponentBasePtr>(J);
      cout << "Built J1" << endl;
      miniqmcreference::OneBodyJastrowRef<BsplineFunctor<RealType>> *J_ref =
          new miniqmcreference::OneBodyJastrowRef<BsplineFunctor<RealType>>(
              ions, els_ref);
      buildJ1(*J_ref, els.Lattice.WignerSeitzRadius);
      wfc_ref = dynamic_cast<WaveFunctionComponentBasePtr>(J_ref);
      cout << "Built J1_ref" << endl;
    }
    else if (wfc_name == "JeeI" || wfc_name == "J3")
    {
      ThreeBodyJastrow<PolynomialFunctor3D> *J =
          new ThreeBodyJastrow<PolynomialFunctor3D>(ions, els);
      buildJeeI(*J, els.Lattice.WignerSeitzRadius);
      wfc = dynamic_cast<WaveFunctionComponentBasePtr>(J);
      cout << "Built JeeI" << endl;
      miniqmcreference::ThreeBodyJastrowRef<PolynomialFunctor3D> *J_ref =
          new miniqmcreference::ThreeBodyJastrowRef<PolynomialFunctor3D>(
              ions, els_ref);
      buildJeeI(*J_ref, els.Lattice.WignerSeitzRadius);
      wfc_ref = dynamic_cast<WaveFunctionComponentBasePtr>(J_ref);
      cout << "Built JeeI_ref" << endl;
    }

    constexpr RealType czero(0);

    // compute distance tables
    els.update();
    els_ref.update();

    {
      els.G = czero;
      els.L = czero;
      wfc->evaluateLog(els, els.G, els.L);

      els_ref.G = czero;
      els_ref.L = czero;
      wfc_ref->evaluateLog(els_ref, els_ref.G, els_ref.L);

      cout << "Check values " << wfc->LogValue << " " << els.G[12] << " "
           << els.L[12] << endl;
      cout << "Check values ref " << wfc_ref->LogValue << " " << els_ref.G[12]
           << " " << els_ref.L[12] << endl
           << endl;
      cout << "evaluateLog::V Error = "
           << (wfc->LogValue - wfc_ref->LogValue) / nels << endl;
      evaluateLog_v_err +=
          std::fabs((wfc->LogValue - wfc_ref->LogValue) / nels);
      {
        double g_err = 0.0;
        for (int iel = 0; iel < nels; ++iel)
        {
          PosType dr = (els.G[iel] - els_ref.G[iel]);
          RealType d = sqrt(dot(dr, dr));
          g_err += d;
        }
        cout << "evaluateLog::G Error = " << g_err / nels << endl;
        evaluateLog_g_err += std::fabs(g_err / nels);
      }
      {
        double l_err = 0.0;
        for (int iel = 0; iel < nels; ++iel)
        {
          l_err += abs(els.L[iel] - els_ref.L[iel]);
        }
        cout << "evaluateLog::L Error = " << l_err / nels << endl;
        evaluateLog_l_err += std::fabs(l_err / nels);
      }

      random_th.generate_normal(&delta[0][0], nels3);
      double g_eval  = 0.0;
      double r_ratio = 0.0;
      double g_ratio = 0.0;

      int naccepted = 0;

      for (int iel = 0; iel < nels; ++iel)
      {
        els.setActive(iel);
        PosType grad_soa = wfc->evalGrad(els, iel);

        els_ref.setActive(iel);
        PosType grad_ref = wfc_ref->evalGrad(els_ref, iel) - grad_soa;
        g_eval += sqrt(dot(grad_ref, grad_ref));

        PosType dr    = sqrttau * delta[iel];
        els.makeMoveAndCheck(iel, dr);
        bool good_ref = els_ref.makeMoveAndCheck(iel, dr);

        if (!good_ref) continue;

        grad_soa       = 0;
        RealType r_soa = wfc->ratioGrad(els, iel, grad_soa);
        grad_ref       = 0;
        RealType r_ref = wfc_ref->ratioGrad(els_ref, iel, grad_ref);

        grad_ref -= grad_soa;
        g_ratio += sqrt(dot(grad_ref, grad_ref));
        r_ratio += abs(r_soa / r_ref - 1);

        if (ur[iel] < r_ref)
        {
          wfc->acceptMove(els, iel);
          els.acceptMove(iel);

          wfc_ref->acceptMove(els_ref, iel);
          els_ref.acceptMove(iel);
          naccepted++;
        }
        else
        {
          els.rejectMove(iel);
          els_ref.rejectMove(iel);
        }
      }
      cout << "Accepted " << naccepted << "/" << nels << endl;
      cout << "evalGrad::G      Error = " << g_eval / nels << endl;
      cout << "ratioGrad::G     Error = " << g_ratio / nels << endl;
      cout << "ratioGrad::Ratio Error = " << r_ratio / nels << endl;
      evalGrad_g_err += std::fabs(g_eval / nels);
      ratioGrad_g_err += std::fabs(g_ratio / nels);
      ratioGrad_r_err += std::fabs(r_ratio / nels);

      // nothing to do with J2 but needs for general cases
      els.donePbyP();
      els_ref.donePbyP();

      els.G = czero;
      els.L = czero;
      wfc->evaluateGL(els, els.G, els.L);

      els_ref.G = czero;
      els_ref.L = czero;
      wfc_ref->evaluateGL(els_ref, els_ref.G, els_ref.L);

      {
        double g_err = 0.0;
        for (int iel = 0; iel < nels; ++iel)
        {
          PosType dr = (els.G[iel] - els_ref.G[iel]);
          RealType d = sqrt(dot(dr, dr));
          g_err += d;
        }
        cout << "evaluteGL::G Error = " << g_err / nels << endl;
        evaluateGL_g_err += std::fabs(g_err / nels);
      }
      {
        double l_err = 0.0;
        for (int iel = 0; iel < nels; ++iel)
        {
          l_err += abs(els.L[iel] - els_ref.L[iel]);
        }
        cout << "evaluteGL::L Error = " << l_err / nels << endl;
        evaluateGL_l_err += std::fabs(l_err / nels);
      }

      // now ratio only
      r_ratio              = 0.0;
      constexpr int nknots = 12;
      int nsphere          = 0;
      for (int iat = 0; iat < nions; ++iat)
      {
        for (int nj = 0, jmax = d_ie->nadj(iat); nj < jmax; ++nj)
        {
          const RealType r = d_ie->distance(iat, nj);
          if (r < Rmax)
          {
            const int iel = d_ie->iadj(iat, nj);
            nsphere++;
            random_th.generate_uniform(&delta[0][0], nknots * 3);
            for (int k = 0; k < nknots; ++k)
            {
              els.makeMoveOnSphere(iel, delta[k]);
              RealType r_soa = wfc->ratio(els, iel);
              els.rejectMove(iel);

              els_ref.makeMoveOnSphere(iel, delta[k]);
              RealType r_ref = wfc_ref->ratio(els_ref, iel);
              els_ref.rejectMove(iel);
              r_ratio += abs(r_soa / r_ref - 1);
            }
          }
        }
      }
      cout << "ratio with SphereMove  Error = " << r_ratio / nsphere
           << " # of moves =" << nsphere << endl;
      ratio_err += std::fabs(r_ratio / (nels * nknots));
    }
  } // end of omp parallel

  int np                   = omp_get_max_threads();
  constexpr RealType small = std::numeric_limits<RealType>::epsilon() * 1e4;
  bool fail                = false;
  cout << std::endl;
  if (evaluateLog_v_err / np > small)
  {
    cout << "Fail in evaluateLog, V error =" << evaluateLog_v_err / np
         << " for " << wfc_name << std::endl;
    fail = true;
  }
  if (evaluateLog_g_err / np > small)
  {
    cout << "Fail in evaluateLog, G error =" << evaluateLog_g_err / np
         << " for " << wfc_name << std::endl;
    fail = true;
  }
  if (evaluateLog_l_err / np > small)
  {
    cout << "Fail in evaluateLog, L error =" << evaluateLog_l_err / np
         << " for " << wfc_name << std::endl;
    fail = true;
  }
  if (evalGrad_g_err / np > small)
  {
    cout << "Fail in evalGrad, G error =" << evalGrad_g_err / np << " for "
         << wfc_name << std::endl;
    fail = true;
  }
  if (ratioGrad_r_err / np > small)
  {
    cout << "Fail in ratioGrad, ratio error =" << ratioGrad_r_err / np
         << " for " << wfc_name << std::endl;
    fail = true;
  }
  if (ratioGrad_g_err / np > small)
  {
    cout << "Fail in ratioGrad, G error =" << ratioGrad_g_err / np << " for "
         << wfc_name << std::endl;
    fail = true;
  }
  if (evaluateGL_g_err / np > small)
  {
    cout << "Fail in evaluateGL, G error =" << evaluateGL_g_err / np
         << " for " << wfc_name << std::endl;
    fail = true;
  }
  if (evaluateGL_l_err / np > small)
  {
    cout << "Fail in evaluateGL, L error =" << evaluateGL_l_err / np
         << " for " << wfc_name << std::endl;
    fail = true;
  }
  if (ratio_err / np > small)
  {
    cout << "Fail in ratio, ratio error =" << ratio_err / np << " for "
         << wfc_name << std::endl;
    fail = true;
  }
  if (!fail) cout << "All checks passed for " << wfc_name << std::endl;

  return 0;
}
