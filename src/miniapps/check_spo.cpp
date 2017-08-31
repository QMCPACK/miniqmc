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
#include <Configuration.h>
#include <Particle/ParticleSet.h>
#include <Utilities/RandomGenerator.h>
#include <Simulation/Simulation.hpp>
#include <miniapps/pseudo.hpp>
#include <spline2/MultiBsplineRef.hpp>
#include <QMCWaveFunctions/einspline_spo.hpp>
#include <getopt.h>

using namespace std;
using namespace qmcplusplus;

int main(int argc, char **argv)
{

  OhmmsInfo("check_spo");

  // clang-format off
  typedef QMCTraits::RealType           RealType;
  typedef ParticleSet::ParticlePos_t    ParticlePos_t;
  typedef ParticleSet::ParticleLayout_t LatticeType;
  typedef ParticleSet::TensorType       TensorType;
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
  int nx = 48, ny = 48, nz = 60;
  int tileSize = -1;

  char *g_opt_arg;
  int opt;
  while ((opt = getopt(argc, argv, "hsg:i:b:c:a:")) != -1)
  {
    switch (opt)
    {
    case 'h': printf("[-g \"n0 n1 n2\"]\n"); return 1;
    case 'g': // tiling1 tiling2 tiling3
      sscanf(optarg, "%d %d %d", &na, &nb, &nc);
      break;
    case 'i': // number of MC steps
      nsteps = atoi(optarg);
      break;
    case 's': // random seed
      iseed = atoi(optarg);
      break;
    case 'a': tileSize = atoi(optarg); break;
    }
  }

  Tensor<int, 3> tmat(na, 0, 0, 0, nb, 0, 0, 0, nc);

  // turn off output
  if (omp_get_max_threads() > 1)
  {
    OhmmsInfo::Log->turnoff();
    OhmmsInfo::Warn->turnoff();
  }

  using spo_type =
      einspline_spo<RealType, MultiBspline<RealType> >;
  spo_type spo_main;
  using spo_ref_type =
      einspline_spo<RealType, MultiBsplineRef<RealType> >;
  spo_ref_type spo_ref_main;
  int nTiles = 1;

  {
    Tensor<RealType, 3> lattice_b;
    ParticleSet ions;
    RealType scale  = 1.0;
    lattice_b       = tile_cell(ions, tmat, scale);
    const int nions = ions.getTotalNum();
    const int norb  = count_electrons(ions, 1) / 2;
    tileSize        = (tileSize > 0) ? tileSize : norb;
    nTiles          = norb / tileSize;
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

  // construct ion particles
  ParticleSet ions;
  const RealType scale = 1.0;
  ions.Lattice.BoxBConds      = 1;
  tile_cell(ions, tmat, scale);
  ions.update();
  const int nions = ions.getTotalNum();
  const int nels  = count_electrons(ions, 1);
  const int nels3 = 3 * nels;

  std::vector<RandomGenerator<RealType> *> rng_list;
  std::vector<ParticleSet *>               els_list;
  std::vector<spo_type *>                  spo_list;
  std::vector<spo_ref_type *>              spo_ref_list;
  std::vector<NonLocalPP<RealType> *>      nlpp_list;

  #pragma omp parallel
  {
    const int np = omp_get_num_threads();
    const int ip = omp_get_thread_num();

    #pragma omp single
    {
      rng_list.resize(np);
      els_list.resize(np);
      spo_list.resize(np);
      spo_ref_list.resize(np);
      nlpp_list.resize(np);
    }

    // create RNG within the thread
    rng_list[ip] = new RandomGenerator<RealType>(MakeSeed(ip, np));
    RandomGenerator<RealType> &random_th = *rng_list[ip];

    // create elecs within the thread
    els_list[ip]     = new ParticleSet;
    ParticleSet &els = *els_list[ip];

    // create up/down electrons
    els.Lattice.BoxBConds = 1;
    els.Lattice.set(ions.Lattice);
    vector<int> ud(2);
    ud[0] = nels / 2;
    ud[1] = nels - ud[0];
    els.create(ud);
    els.R.InUnit = 1;
    random_th.generate_uniform(&els.R[0][0], nels3);
    els.convert2Cart(els.R); // convert to Cartiesian
    // update content: compute distance tables and structure factor
    els.update();

    // create spo per thread
    spo_list[ip] = new spo_type(spo_main, 1, 0);
    spo_type &spo = *spo_list[ip];
    spo_ref_list[ip] = new spo_ref_type(spo_ref_main, 1, 0);
    spo_ref_type &spo_ref = *spo_ref_list[ip];

    // create pseudopp per thread
    nlpp_list[ip]             = new NonLocalPP<RealType>(random_th);
    NonLocalPP<RealType> &ecp = *nlpp_list[ip];
  }

  double ratio         = 0.0;
  double nspheremoves  = 0.0;
  double dNumVGHCalls  = 0.0;
  double evalV_v_err   = 0.0;
  double evalVGH_v_err = 0.0;
  double evalVGH_g_err = 0.0;
  double evalVGH_h_err = 0.0;

// clang-format off
  #pragma omp parallel for reduction(+:ratio,nspheremoves,dNumVGHCalls) \
   reduction(+:evalV_v_err,evalVGH_v_err,evalVGH_g_err,evalVGH_h_err)
  // clang-format on
  for(size_t iw=0; iw<els_list.size(); iw++)
  {
    // load a walker
    RandomGenerator<RealType> &random_th = *rng_list[iw];
    ParticleSet                     &els = *els_list[iw];
    spo_type                        &spo = *spo_list[iw];
    spo_ref_type                &spo_ref = *spo_ref_list[iw];
    NonLocalPP<RealType>            &ecp = *nlpp_list[iw];

    // this is the cutoff from the non-local PP
    const RealType Rmax(1.7);
    const int nknots(ecp.size());
    const RealType tau = 2.0;

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
  if (!fail) cout << "All checking pass!" << std::endl;

  return 0;
}
