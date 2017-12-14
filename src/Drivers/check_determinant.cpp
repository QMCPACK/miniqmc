////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by: M. Graham Lopez, Oak Ridge National Lab
//
// File created by: M. Graham Lopez, Oak Ridge National Lab
////////////////////////////////////////////////////////////////////////////////
// -*- C++ -*-
// clang-format off
/** @file check_determinant.cpp


  Compares against a reference implementation for correctness.

 */

// clang-format on

#include <Utilities/Configuration.h>
#include <Particle/ParticleSet.h>
#include <Utilities/PrimeNumberSet.h>
#include <Utilities/RandomGenerator.h>
#include <Utilities/qmcpack_version.h>
#include <Input/Input.hpp>
#include <QMCWaveFunctions/Determinant.h>
#include <QMCWaveFunctions/DeterminantRef.h>
#include <getopt.h>

using namespace std;
using namespace qmcplusplus;

void print_help()
{
  //FIXME
}


int main(int argc, char **argv)
{

  OhmmsInfo("miniqmc");

  // clang-format off
  typedef QMCTraits::RealType           RealType;
  typedef ParticleSet::ParticlePos_t    ParticlePos_t;
  typedef ParticleSet::PosType          PosType;
  // clang-format on

  // use the global generator

  // bool ionode=(mycomm->rank() == 0);
  bool ionode = 1;
  int nsteps  = 100;
  int iseed   = 11;
  int nsubsteps = 1;
  // Set cutoff for NLPP use.

  PrimeNumberSet<uint32_t> myPrimes;

  bool verbose = false;

  int opt;
  while ((opt = getopt(argc, argv, "hdvVs:g:i:b:c:a:r:")) != -1)
  {
    switch (opt)
    {
    case 'h': print_help(); return 1;
    case 'i': // number of MC steps
      nsteps = atoi(optarg);
      break;
    case 's': // the number of sub steps for drift/diffusion
      nsubsteps = atoi(optarg);
      break;
    case 'v': verbose  = true; break;
    case 'V':
      print_version(true);
      return 1;
      break;
    }
  }

  Random.init(0, 1, iseed);

  print_version(verbose);

  // turn off output
  if (!verbose || omp_get_max_threads() > 1)
  {
    OhmmsInfo::Log->turnoff();
    OhmmsInfo::Warn->turnoff();
  }

#pragma omp parallel
  {
    ParticleSet ions, els;
    ions.setName("ion");
    els.setName("e");

    const int ip = omp_get_thread_num();

    // create generator within the thread
    RandomGenerator<RealType> random_th(myPrimes[ip]);

    ions.Lattice.BoxBConds = 1;
    ions.RSoA = ions.R; // fill the SoA

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

    miniqmcreference::DiracDeterminantRef determinant_ref(nels, random_th);
    DiracDeterminant determinant(nels, random_th);

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
          PosType dr = sqrttau * delta[iel];
          bool isValid = els.makeMoveAndCheck(iel, dr);

          if (!isValid) continue;

          // Compute gradient at the trial position

          determinant_ref.ratio(iel);
          determinant.ratio(iel);

          // Accept/reject the trial move
          if (ur[iel] > accept) // MC
          {
            // Update position, and update temporary storage
            els.acceptMove(iel);
            determinant_ref.acceptMove(iel);
            determinant.acceptMove(iel);
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
  } // end of omp parallel

  if (ionode)
  {

  }

  return 0;
}
