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
#include <Utilities/Configuration.h>
#include <Particle/ParticleSet.h>
#include <Utilities/RandomGenerator.h>
#include <Input/Input.hpp>
#include <Numerics/Spline2/MultiBsplineRef.hpp>
#include <QMCWaveFunctions/einspline_spo.hpp>
#include <Utilities/qmcpack_version.h>
#include <getopt.h>

using namespace std;
using namespace qmcplusplus;

void print_help()
{
  //clang-format off
  cout << "usage:" << '\n';
  cout << "  check_spo [-hvV] [-g \"n0 n1 n2\"] [-n steps]"             << '\n';
  cout << "             [-r rmax] [-s seed]"                            << '\n';
  cout << "options:"                                                    << '\n';
  cout << "  -g  set the 3D tiling.             default: 1 1 1"         << '\n';
  cout << "  -h  print help and exit"                                   << '\n';
  cout << "  -n  number of MC steps             default: 100"           << '\n';
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

  // bool ionode=(mycomm->rank() == 0);
  bool ionode = 1;
  int na      = 1;
  int nb      = 1;
  int nc      = 1;
  int nsteps  = 100;
  int iseed   = 11;
  RealType Rmax(1.7);
  int nx = 37, ny = 37, nz = 37;
  // thread blocking
  // int team_size=1; //default is 1
  int tileSize = -1;
  int team_size   = 1;

  bool verbose = false;

  int opt;
  while(optind < argc)
  {
    if ((opt = getopt(argc, argv, "hvVa:c:f:g:n:r:s:")) != -1)
    {
      switch (opt)
      {
      case 'a': tileSize = atoi(optarg); break;
      case 'c': // number of members per team
        team_size = atoi(optarg);
        break;
      case 'g': // tiling1 tiling2 tiling3
        sscanf(optarg, "%d %d %d", &na, &nb, &nc);
        break;
      case 'h': print_help(); break;
      case 'n':
        nsteps = atoi(optarg);
        break;
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

  Random.init(0, 1, iseed);
  Tensor<int, 3> tmat(na, 0, 0, 0, nb, 0, 0, 0, nc);

  // turn off output
  if (omp_get_max_threads() > 1)
  {
    outputManager.shutOff();
  }

  OHMMS_PRECISION ratio = 0.0;

  using spo_type =
      einspline_spo<OHMMS_PRECISION, MultiBspline<OHMMS_PRECISION>>;
  spo_type spo_main;
  using spo_ref_type =
      einspline_spo<OHMMS_PRECISION,
                    miniqmcreference::MultiBsplineRef<OHMMS_PRECISION>>;
  spo_ref_type spo_ref_main;
  int nTiles = 1;

  {
    Tensor<OHMMS_PRECISION, 3> lattice_b;
    ParticleSet ions;
    OHMMS_PRECISION scale = 1.0;
    lattice_b             = tile_cell(ions, tmat, scale);
    const int norb        = count_electrons(ions, 1) / 2;
    tileSize              = (tileSize > 0) ? tileSize : norb;
    nTiles                = norb / tileSize;
    if (ionode)
      cout << "\nNumber of orbitals/splines = " << norb
           << " and Tile size = " << tileSize
           << " and Number of tiles = " << nTiles
           << " and Iterations = " << nsteps << endl;
    spo_main.set(nx, ny, nz, norb, nTiles);
    spo_main.Lattice.set(lattice_b);
    spo_ref_main.set(nx, ny, nz, norb, nTiles);
    spo_ref_main.Lattice.set(lattice_b);
  }

  double nspheremoves = 0;
  double dNumVGHCalls = 0;

  double evalV_v_err   = 0.0;
  double evalVGH_v_err = 0.0;
  double evalVGH_g_err = 0.0;
  double evalVGH_h_err = 0.0;

// clang-format off
  #pragma omp parallel reduction(+:ratio,nspheremoves,dNumVGHCalls) \
   reduction(+:evalV_v_err,evalVGH_v_err,evalVGH_g_err,evalVGH_h_err)
  // clang-format on
  {
    const int np     = omp_get_num_threads();
    const int ip     = omp_get_thread_num();
    const int team_id = ip / team_size;
    const int member_id = ip % team_size;

    // create generator within the thread
    RandomGenerator<RealType> random_th(MakeSeed(team_id, np));

    ParticleSet ions, els;
    const OHMMS_PRECISION scale = 1.0;
    ions.Lattice.BoxBConds      = 1;
    tile_cell(ions, tmat, scale);

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
    }

    // update content: compute distance tables and structure factor
    els.update();
    ions.update();

    // create pseudopp
    NonLocalPP<OHMMS_PRECISION> ecp(random_th);
    // create spo per thread
    spo_type spo(spo_main, team_size, member_id);
    spo_ref_type spo_ref(spo_ref_main, team_size, member_id);

    // use teams
    // if(team_size>1 && team_size>=nTiles ) spo.set_range(team_size,ip%team_size);

    // this is the cutoff from the non-local PP
    const int nknots(ecp.size());

    ParticlePos_t delta(nels);
    ParticlePos_t rOnSphere(nknots);

    RealType sqrttau = 2.0;
    RealType accept  = 0.5;

    vector<RealType> ur(nels);
    random_th.generate_uniform(ur.data(), nels);
    const double zval =
        1.0 * static_cast<double>(nels) / static_cast<double>(nions);

    int my_accepted = 0, my_vals = 0;

    for (int mc = 0; mc < nsteps; ++mc)
    {
      random_th.generate_normal(&delta[0][0], nels3);
      random_th.generate_uniform(ur.data(), nels);

      // VMC
      for (int iel = 0; iel < nels; ++iel)
      {
        PosType pos = els.R[iel] + sqrttau * delta[iel];
        spo.evaluate_vgh(pos);
        spo_ref.evaluate_vgh(pos);
        // accumulate error
        for (int ib = 0; ib < spo.nBlocks; ib++)
          for (int n = 0; n < spo.nSplinesPerBlock; n++)
          {
            // value
            evalVGH_v_err +=
                std::fabs((*spo.psi[ib])[n] - (*spo_ref.psi[ib])[n]);
            // grad
            evalVGH_g_err += std::fabs(spo.grad[ib]->data(0)[n] -
                                       spo_ref.grad[ib]->data(0)[n]);
            evalVGH_g_err += std::fabs(spo.grad[ib]->data(1)[n] -
                                       spo_ref.grad[ib]->data(1)[n]);
            evalVGH_g_err += std::fabs(spo.grad[ib]->data(2)[n] -
                                       spo_ref.grad[ib]->data(2)[n]);
            // hess
            evalVGH_h_err += std::fabs(spo.hess[ib]->data(0)[n] -
                                       spo_ref.hess[ib]->data(0)[n]);
            evalVGH_h_err += std::fabs(spo.hess[ib]->data(1)[n] -
                                       spo_ref.hess[ib]->data(1)[n]);
            evalVGH_h_err += std::fabs(spo.hess[ib]->data(2)[n] -
                                       spo_ref.hess[ib]->data(2)[n]);
            evalVGH_h_err += std::fabs(spo.hess[ib]->data(3)[n] -
                                       spo_ref.hess[ib]->data(3)[n]);
            evalVGH_h_err += std::fabs(spo.hess[ib]->data(4)[n] -
                                       spo_ref.hess[ib]->data(4)[n]);
            evalVGH_h_err += std::fabs(spo.hess[ib]->data(5)[n] -
                                       spo_ref.hess[ib]->data(5)[n]);
          }
        if (ur[iel] > accept)
        {
          els.R[iel] = pos;
          my_accepted++;
        }
      }

      random_th.generate_uniform(ur.data(), nels);
      ecp.randomize(rOnSphere); // pick random sphere
      for (int iat = 0, kat = 0; iat < nions; ++iat)
      {
        const int nnF = static_cast<int>(ur[kat++] * zval);
        RealType r    = Rmax * ur[kat++];
        auto centerP  = ions.R[iat];
        my_vals += (nnF * nknots);

        for (int nn = 0; nn < nnF; ++nn)
        {
          for (int k = 0; k < nknots; k++)
          {
            PosType pos = centerP + r * rOnSphere[k];
            spo.evaluate_v(pos);
            spo_ref.evaluate_v(pos);
            // accumulate error
            for (int ib = 0; ib < spo.nBlocks; ib++)
              for (int n = 0; n < spo.nSplinesPerBlock; n++)
                evalV_v_err +=
                    std::fabs((*spo.psi[ib])[n] - (*spo_ref.psi[ib])[n]);
          }
        } // els
      }   // ions

    } // steps.

    ratio += RealType(my_accepted) / RealType(nels * nsteps);
    nspheremoves += RealType(my_vals) / RealType(nsteps);
    dNumVGHCalls += nels;

  } // end of omp parallel

  evalV_v_err /= nspheremoves;
  evalVGH_v_err /= dNumVGHCalls;
  evalVGH_g_err /= dNumVGHCalls;
  evalVGH_h_err /= dNumVGHCalls;

  int np                     = omp_get_max_threads();
  constexpr RealType small_v = std::numeric_limits<RealType>::epsilon() * 1e4;
  constexpr RealType small_g = std::numeric_limits<RealType>::epsilon() * 3e6;
  constexpr RealType small_h = std::numeric_limits<RealType>::epsilon() * 6e8;
  bool fail                  = false;
  cout << std::endl;
  if (evalV_v_err / np > small_v)
  {
    cout << "Fail in evaluate_v, V error =" << evalV_v_err / np << std::endl;
    fail = true;
  }
  if (evalVGH_v_err / np > small_v)
  {
    cout << "Fail in evaluate_vgh, V error =" << evalVGH_v_err / np
         << std::endl;
    fail = true;
  }
  if (evalVGH_g_err / np > small_g)
  {
    cout << "Fail in evaluate_vgh, G error =" << evalVGH_g_err / np
         << std::endl;
    fail = true;
  }
  if (evalVGH_h_err / np > small_h)
  {
    cout << "Fail in evaluate_vgh, H error =" << evalVGH_h_err / np
         << std::endl;
    fail = true;
  }
  if (!fail) cout << "All checks passed for spo" << std::endl;

  return 0;
}
