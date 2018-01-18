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
// clang-format off
/** @file miniqmc.cpp
    @brief Miniapp to capture the computation in particle moves.

 @mainpage MiniQMC: miniapp for QMCPACK kernels

 Implemented kernels
   - \subpage JastrowFactors "Jastrow Factors" includes one-body, two-body and three-body Jastrow
     factors.
   - \subpage SPO "Single Particle Orbitals" (SPO) based on splines
   - \subpage InverseUpdate "Inverse matrix update" for determinant
   - \subpage ParticleHandling "Particle distances" and boundary conditions



  The \ref src/Drivers/miniqmc.cpp "miniqmc" driver models particle moves and evaluation of the wavefunction.
  The <a href="https://github.com/QMCPACK/miniqmc/wiki#miniqmc-computational-overview">wiki</a> gives an outline of the computation.

  The \ref src/Drivers/check_wfc.cpp "check_wfc", \ref src/Drivers/check_spo.cpp "check_spo",
  and \ref src/Drivers/check_determinant.cpp "check_determinant" drivers check correctness by comparing against
  reference implementations of the Jastrow, SPO, and determinant inverse (respectively).
  The code for the reference implementation uses a `Ref` suffix and is contained in the \ref miniqmcreference namespace.


 */

 /*!
 \page JastrowFactors Jastrow Factors

  The Jastrow factor accounts for the correlation of electron-ion pairs
  (one-body Jastrow), two-electron pairs (two-body Jastrow) and
  two-electron-one-ion trios (three-body/eeI Jastrow).

  The Jastrow factor is composed from two types of classes - the first is for
  the types of particles involved (one/two/three body), and the second is the
  functional form for the radial part.  The classes for the first part are
  qmcplusplus::OneBodyJastrow, qmcplusplus::TwoBodyJastrow and
  qmcplusplus::ThreeBodyJastrow.  The second part uses 1D B-splines, defined
  in qmcplusplus::BsplineFunctor, for one and two body Jastrow and polynomials,
  defined in qmcplusplus::PolynomialFunctor3D, for three body Jastrow.

  This miniapp only contains the B-spline and polynomial functional form, since
  it is the most widely used.  The QMCPACK distribution contains other
  functional forms.
 */

 /*!
 \page SPO Single Particle Orbitals

  The Single Particle Orbitals (SPO) depend only on individual electron coordinates and are represented by a 3D spline.
  The 3D spline code is located in \ref src/Numerics/Spline2, with the evaluation code in the \ref qmcplusplus::MultiBspline "MultiBspline" class.
  The connection from the wavefunction to the spline functions is located in the \ref qmcplusplus::einspline_spo "einspline_spo" class.

  The core evaluation routine evaluates the value, the gradient, and the Laplacian at a given electron coordinate.

  The size of the coefficient data set can be large - on the order of gigabytes.

 */

 /*!
 \page InverseUpdate Inverse Matrix Update

  The inverse matrix and updating is handled by \ref qmcplusplus::DiracDeterminant "DiracDeterminant".
  The initial creation and inversion of the matrix occurs in qmcplusplus::DiracDeterminant::recompute, which is called on the first
  call to qmcplusplus::DiracDeterminant::evaluateLog.   The rows are updated after accepted Monte Carlo moves
  via updateRow (in \ref src/QMCWaveFunctions/Determinant.h), which is called from qmcplusplus::DiracDeterminant::acceptMove.

  The implementation for updateRow is
  \snippet QMCWaveFunctions/Determinant.h UpdateRow
 */

 /*!
 \page ParticleHandling Particle positions, distances, and boundary conditions

  The qmcpluplus:ParticleSet class holds particle positions, lattice information, and distance tables.
  The positions are stored in the \ref qmcplusplus::ParticleSet#R member, and a copy is kept in a Structure-of-Arrays (SoA) layout in
  \ref qmcplusplus::ParticleSet#RSoA.

  Distances, using the minimum image conventions with periodic boundaries, are computed in \ref src/Particle/Lattice/ParticleBConds.h.

  Distances are stored in distance tables, where qmcplusplus::DistanceTableData is the base class for the storage.  There are two types
  of distance tables.  One is for similar particles (qmcplusplus::DistanceTableAA), such as electron-electron distances.  The other
  is for dissimilar particles (qmcplusplus::DistanceTableBA), such as electron-ion distances.
 */

// clang-format on

#include <Utilities/Configuration.h>
#include <Particle/ParticleSet.h>
#include <Particle/DistanceTable.h>
#include <Utilities/PrimeNumberSet.h>
#include <Utilities/NewTimer.h>
#include <Utilities/RandomGenerator.h>
#include <Utilities/qmcpack_version.h>
#include <Input/Input.hpp>
#include <QMCWaveFunctions/einspline_spo.hpp>
#include <QMCWaveFunctions/WaveFunction.h>
#include <getopt.h>

#ifdef QMC_USE_KOKKOS
#include <Kokkos_Core.hpp>
#endif

using namespace std;
using namespace qmcplusplus;

enum MiniQMCTimers
{
  Timer_Total,
  Timer_Diffusion,
  Timer_GL,
  Timer_ECP,
  Timer_Value,
  Timer_evalGrad,
  Timer_ratioGrad,
  Timer_Update,
  Timer_Wavefunction,
  Timer_DT,
  Timer_SPO
};

TimerNameList_t<MiniQMCTimers> MiniQMCTimerNames = {
    {Timer_Total, "Total"},
    {Timer_Diffusion, "Diffusion"},
    {Timer_GL, "Wavefuntion GL"},
    {Timer_ECP, "Pseudopotential"},
    {Timer_Value, "Value"},
    {Timer_evalGrad, "Current Gradient"},
    {Timer_ratioGrad, "New Gradient"},
    {Timer_Update, "Update"},
    {Timer_Wavefunction, "Wavefunction"},
    {Timer_SPO, "Single-Particle Orbitals"},
    {Timer_DT, "Distance Tables"},
};

void print_help()
{
  //clang-format off
  cout << "usage:" << '\n';
  cout << "  miniqmc   [-hvV] [-g \"n0 n1 n2\"] [-n steps]"             << '\n';
  cout << "            [-N substeps] [-r rmax] [-s seed]"               << '\n';
  cout << "options:"                                                    << '\n';
  cout << "  -g  set the 3D tiling.             default: 1 1 1"         << '\n';
  cout << "  -h  print help and exit"                                   << '\n';
  cout << "  -n  number of MC steps             default: 100"           << '\n';
  cout << "  -N  number of MC substeps          default: 1"             << '\n';
  cout << "  -r  set the Rmax.                  default: 1.7"           << '\n';
  cout << "  -s  set the random seed.           default: 11"            << '\n';
  cout << "  -v  verbose output"                                        << '\n';
  cout << "  -V  print version information and exit"                    << '\n';
  //clang-format on

  exit(1); // print help and exit
}

int main(int argc, char **argv)
{

#ifdef QMC_USE_KOKKOS
  Kokkos::initialize(argc, argv);
#endif

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
  int nx = 37, ny = 37, nz = 37;
  // thread blocking
  // int team_size=1; //default is 1
  int tileSize  = -1;
  int team_size = 1;
  int nsubsteps = 1;
  // Set cutoff for NLPP use.
  RealType Rmax(1.7);
  bool useRef = false;

  PrimeNumberSet<uint32_t> myPrimes;

  bool verbose = false;

  int opt;
  while(optind < argc)
  {
    if ((opt = getopt(argc, argv, "hvVa:c:g:n:N:r:s:")) != -1)
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
      case 'N':
        nsubsteps = atoi(optarg);
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

  Random.init(0, 1, iseed);
  Tensor<int, 3> tmat(na, 0, 0, 0, nb, 0, 0, 0, nc);

  TimerManager.set_timer_threshold(timer_level_coarse);
  TimerList_t Timers;
  setup_timers(Timers, MiniQMCTimerNames, timer_level_coarse);

  print_version(verbose);

  if (verbose) {
    outputManager.setVerbosity(Verbosity::HIGH);
  }

  // turn off output
  if (!verbose || omp_get_max_threads() > 1)
  {
    outputManager.shutOff();
  }

  int nthreads = omp_get_max_threads();

  using spo_type = einspline_spo<OHMMS_PRECISION>;
  spo_type spo_main;
  int nTiles = 1;

  // Temporally create ParticleSet ions for setting splines.
  // Per-thread ions will be created later to avoid any performance impact from
  // shared ions.
  {
    Tensor<OHMMS_PRECISION, 3> lattice_b;
    ParticleSet ions;
    OHMMS_PRECISION scale = 1.0;
    lattice_b             = tile_cell(ions, tmat, scale);
    const int nels        = count_electrons(ions, 1);
    const int norb        = nels / 2;
    tileSize              = (tileSize > 0) ? tileSize : norb;
    nTiles                = norb / tileSize;

    const unsigned int SPO_coeff_size =
        (nx + 3) * (ny + 3) * (nz + 3) * norb * sizeof(RealType);
    double SPO_coeff_size_MB = SPO_coeff_size * 1.0 / 1024 / 1024;
    if (ionode)
    {
      cout << "\nNumber of orbitals/splines = " << norb << endl;
      cout << "Tile size = " << tileSize << endl;
      cout << "Number of tiles = " << nTiles << endl;
      cout << "Number of electrons = " << nels << endl;
      cout << "Iterations = " << nsteps << endl;
      cout << "Rmax " << Rmax << endl;
      cout << "OpenMP threads " << nthreads << endl;

      cout << "\nSPO coefficients size = " << SPO_coeff_size;
      cout << " bytes (" << SPO_coeff_size_MB << " MB)" << endl;
    }
    spo_main.set(nx, ny, nz, norb, nTiles);
    spo_main.Lattice.set(lattice_b);
  }

  if (ionode)
  {
    if (!useRef)
      cout << "Using SoA distance table, Jastrow + einspline, " << endl
           << "and determinant update." << endl;
    else
      cout << "Using the reference implementation for Jastrow, " << endl
           << "determinant update, and distance table + einspline of the "
           << endl
           << "reference implementation " << endl;
  }

  Timers[Timer_Total]->start();
  //#pragma omp parallel
  {
    ParticleSet ions, els;
    ions.setName("ion");
    els.setName("e");

    const int ip = omp_get_thread_num();

    const int member_id = ip % team_size;

    // create spo per thread
    spo_type spo(spo_main, team_size, member_id);

    // create generator within the thread
    RandomGenerator<RealType> random_th(myPrimes[ip]);

    ions.Lattice.BoxBConds = 1;
    OHMMS_PRECISION scale  = 1.0;
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

    WaveFunctionBase *wavefunction;

    if (useRef)
    {
      wavefunction =
          new miniqmcreference::WaveFunctionRef(ions, els, random_th);
    }
    else
    {
      wavefunction = new WaveFunction(ions, els, random_th);
    }

    // set Rmax for ion-el distance table for PP
    wavefunction->setRmax(Rmax);

    // create pseudopp
    NonLocalPP<OHMMS_PRECISION> ecp(random_th);

    // this is the cutoff from the non-local PP
    const int nknots(ecp.size());

    // For VMC, tau is large and should result in an acceptance ratio of roughly
    // 50%
    // For DMC, tau is small and should result in an acceptance ratio of 99%
    const RealType tau = 2.0;

    ParticlePos_t delta(nels);
    ParticlePos_t rOnSphere(nknots);

    RealType sqrttau = std::sqrt(tau);
    RealType accept  = 0.5;

    aligned_vector<RealType> ur(nels);
    random_th.generate_uniform(ur.data(), nels);

    els.update();
    wavefunction->evaluateLog(els);

    int my_accepted = 0;
    for (int mc = 0; mc < nsteps; ++mc)
    {
      Timers[Timer_Diffusion]->start();
      for (int l = 0; l < nsubsteps; ++l) // drift-and-diffusion
      {
        random_th.generate_normal(&delta[0][0], nels3);
        for (int iel = 0; iel < nels; ++iel)
        {
          // Operate on electron with index iel
          Timers[Timer_DT]->start();
          els.setActive(iel);
          Timers[Timer_DT]->stop();
          // Compute gradient at the current position
          Timers[Timer_evalGrad]->start();
          Timers[Timer_Wavefunction]->start();
          PosType grad_now = wavefunction->evalGrad(els, iel);
          Timers[Timer_Wavefunction]->stop();
          Timers[Timer_evalGrad]->stop();

          // Construct trial move
          PosType dr = sqrttau * delta[iel];
          Timers[Timer_DT]->start();
          bool isValid = els.makeMoveAndCheck(iel, dr);
          Timers[Timer_DT]->stop();

          if (!isValid) continue;

          // Compute gradient at the trial position
          Timers[Timer_ratioGrad]->start();

          Timers[Timer_Wavefunction]->start();
          PosType grad_new;
          wavefunction->ratioGrad(els, iel, grad_new);
          Timers[Timer_Wavefunction]->stop();

          Timers[Timer_SPO]->start();
          spo.evaluate_vgh(els.R[iel]);
          Timers[Timer_SPO]->stop();

          Timers[Timer_ratioGrad]->stop();

          // Accept/reject the trial move
          if (ur[iel] > accept) // MC
          {
            // Update position, and update temporary storage
            Timers[Timer_Update]->start();
            wavefunction->acceptMove(els, iel);
            Timers[Timer_Update]->stop();
            Timers[Timer_DT]->start();
            els.acceptMove(iel);
            Timers[Timer_DT]->stop();
            my_accepted++;
          }
          else
          {
            els.rejectMove(iel);
            wavefunction->restore(iel);
          }
        } // iel
      }   // substeps

      Timers[Timer_DT]->start();
      els.donePbyP();
      Timers[Timer_DT]->stop();

      // evaluate Kinetic Energy
      Timers[Timer_GL]->start();
      wavefunction->evaluateGL(els);
      Timers[Timer_GL]->stop();

      Timers[Timer_Diffusion]->stop();

      // Compute NLPP energy using integral over spherical points

      ecp.randomize(rOnSphere); // pick random sphere
      const DistanceTableData *d_ie = wavefunction->d_ie;

      Timers[Timer_ECP]->start();
      for (int iat = 0; iat < nions; ++iat)
      {
        const auto centerP = ions.R[iat];
        for (int nj = 0, jmax = d_ie->nadj(iat); nj < jmax; ++nj)
        {
          const auto r = d_ie->distance(iat, nj);
          if (r < Rmax)
          {
            const int iel = d_ie->iadj(iat, nj);
            const auto dr = d_ie->displacement(iat, nj);
            for (int k = 0; k < nknots; k++)
            {
              PosType deltar(r * rOnSphere[k] - dr);

              Timers[Timer_DT]->start();
              els.makeMoveOnSphere(iel, deltar);
              Timers[Timer_DT]->stop();

              Timers[Timer_Value]->start();

              Timers[Timer_SPO]->start();
              spo.evaluate_v(els.R[iel]);
              Timers[Timer_SPO]->stop();

              Timers[Timer_Wavefunction]->start();
              wavefunction->ratio(els, iel);
              Timers[Timer_Wavefunction]->stop();

              Timers[Timer_Value]->stop();

              els.rejectMove(iel);
            }
          }
        }
      }
      Timers[Timer_ECP]->stop();

    } // nsteps

    // cleanup
    delete wavefunction;
  } // end of omp parallel
  Timers[Timer_Total]->stop();

  if (ionode)
  {
    cout << "================================== " << endl;

    TimerManager.print();
  }

#ifdef QMC_USE_KOKKOS
  Kokkos::finalize();
#endif

  return 0;
}
