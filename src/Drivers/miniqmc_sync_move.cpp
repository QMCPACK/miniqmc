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
#include <Utilities/Communicate.h>
#include <Particle/ParticleSet.h>
#include <Particle/DistanceTable.h>
#include <Utilities/PrimeNumberSet.h>
#include <Utilities/NewTimer.h>
#include <Utilities/XMLWriter.h>
#include <Utilities/RandomGenerator.h>
#include <Utilities/qmcpack_version.h>
#include <Input/Input.hpp>
#include <QMCWaveFunctions/SPOSet.h>
#include <QMCWaveFunctions/SPOSet_builder.h>
#include <QMCWaveFunctions/WaveFunction.h>
#include <Drivers/Mover.hpp>
#include <getopt.h>

using namespace std;
using namespace qmcplusplus;

enum MiniQMCTimers
{
  Timer_Total,
  Timer_Init,
  Timer_Diffusion,
  Timer_ECP,
  Timer_Value,
  Timer_evalGrad,
  Timer_ratioGrad,
  Timer_Update,
  Timer_Setup,
};

TimerNameList_t<MiniQMCTimers> MiniQMCTimerNames = {
    {Timer_Total, "Total"},
    {Timer_Init, "Initialization"},
    {Timer_Diffusion, "Diffusion"},
    {Timer_ECP, "Pseudopotential"},
    {Timer_Value, "Value"},
    {Timer_evalGrad, "Current Gradient"},
    {Timer_ratioGrad, "New Gradient"},
    {Timer_Update, "Update"},
    {Timer_Setup, "Setup"},
};

void print_help()
{
  // clang-format off
  app_summary() << "usage:" << '\n';
  app_summary() << "  miniqmc   [-bhjPvV] [-g \"n0 n1 n2\"] [-m meshfactor]"     << '\n';
  app_summary() << "            [-n steps] [-N substeps] [-x rmax]"              << '\n';
  app_summary() << "            [-r AcceptanceRatio] [-s seed] [-w walkers]"     << '\n';
  app_summary() << "            [-a tile_size] [-t timer_level] [-B nw_b]"       << '\n';
  app_summary() << "            [-u delay_rank]"                                 << '\n';
  app_summary() << "options:"                                                    << '\n';
  app_summary() << "  -a  size of each spline tile       default: num of orbs"   << '\n';
  app_summary() << "  -b  use reference implementations  default: off"           << '\n';
  app_summary() << "  -B  number of walkers per batch    default: 1"             << '\n';
  app_summary() << "  -g  set the 3D tiling.             default: 1 1 1"         << '\n';
  app_summary() << "  -h  print help and exit"                                   << '\n';
  app_summary() << "  -j  enable three body Jastrow      default: off"           << '\n';
  app_summary() << "  -m  meshfactor                     default: 1.0"           << '\n';
  app_summary() << "  -n  number of MC steps             default: 5"             << '\n';
  app_summary() << "  -N  number of MC substeps          default: 1"             << '\n';
  app_summary() << "  -P  not running pseudo potential   default: off"           << '\n';
  app_summary() << "  -r  set the acceptance ratio.      default: 0.5"           << '\n';
  app_summary() << "  -s  set the random seed.           default: 11"            << '\n';
  app_summary() << "  -t  timer level: coarse or fine    default: fine"          << '\n';
  app_summary() << "  -u  matrix delayed update rank     default: 32"            << '\n';
  app_summary() << "  -v  verbose output"                                        << '\n';
  app_summary() << "  -V  print version information and exit"                    << '\n';
  app_summary() << "  -w  number of walker(movers)       default: num of threads"<< '\n';
  app_summary() << "  -x  set the Rmax.                  default: 1.7"           << '\n';
  // clang-format on
}

int main(int argc, char** argv)
{
  // clang-format off
  typedef QMCTraits::RealType           RealType;
  typedef ParticleSet::PosType          PosType;
  typedef ParticleSet::GradType         GradType;
  typedef ParticleSet::ValueType        ValueType;
  // clang-format on

  Communicate comm(argc, argv);

  // use the global generator

  int na     = 1;
  int nb     = 1;
  int nc     = 1;
  int nsteps = 5;
  int iseed  = 11;
  int nx = 37, ny = 37, nz = 37;
  int nmovers = -1;
  // number of walkers per batch
  int nw_b = 1;
  // thread blocking
  int tileSize  = -1;
  int team_size = 1;
  int nsubsteps = 1;
  // Set cutoff for NLPP use.
  RealType Rmax(1.7);
  RealType accept  = 0.5;
  int delay_rank = 32;
  bool useRef   = false;
  bool enableJ3 = false;
  bool run_pseudo = true;

  PrimeNumberSet<uint32_t> myPrimes;

  bool verbose                 = false;
  std::string timer_level_name = "fine";

  if (!comm.root())
  {
    outputManager.shutOff();
  }

  int opt;
  while (optind < argc)
  {
    if ((opt = getopt(argc, argv, "bhjPvVa:B:c:g:m:n:N:r:s:t:u:w:x:")) != -1)
    {
      switch (opt)
      {
      case 'a':
        tileSize = atoi(optarg);
        break;
      case 'b':
        useRef = true;
        break;
      case 'B': // number of walkers per batch
        nw_b = atoi(optarg);
        break;
      case 'c': // number of members per team
        team_size = atoi(optarg);
        break;
      case 'g': // tiling1 tiling2 tiling3
        sscanf(optarg, "%d %d %d", &na, &nb, &nc);
        break;
      case 'h':
        print_help();
        return 1;
        break;
      case 'j':
        enableJ3 = true;
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
      case 'N':
        nsubsteps = atoi(optarg);
        break;
      case 'P':
        run_pseudo = false;
        break;
      case 'r':
        accept = atof(optarg);
        break;
      case 's':
        iseed = atoi(optarg);
        break;
      case 't':
        timer_level_name = std::string(optarg);
        break;
      case 'u':
        delay_rank = atoi(optarg);
        break;
      case 'v':
        verbose = true;
        break;
      case 'V':
        print_version(true);
        return 1;
        break;
      case 'w': // number of nmovers
        nmovers = atoi(optarg);
        break;
      case 'x': // rmax
        Rmax = atof(optarg);
        break;
      default:
        print_help();
        return 1;
      }
    }
    else // disallow non-option arguments
    {
      app_error() << "Non-option arguments not allowed" << endl;
      print_help();
    }
  }

  // set default number of walkers
  if(nmovers<0) nmovers = omp_get_max_threads() * nw_b;
  // cap nw_b
  if(nw_b>nmovers) nw_b = nmovers;
  // number of batches
  const int nbatches = (nmovers+nw_b-1)/nw_b;

  int number_of_electrons = 0;

  Tensor<int, 3> tmat(na, 0, 0, 0, nb, 0, 0, 0, nc);

  timer_levels timer_level = timer_level_fine;
  if (timer_level_name == "coarse")
  {
    timer_level = timer_level_coarse;
  }
  else if (timer_level_name != "fine")
  {
    app_error() << "Timer level should be 'coarse' or 'fine', name given: " << timer_level_name
                << endl;
    return 1;
  }

  TimerManager.set_timer_threshold(timer_level);
  TimerList_t Timers;
  setup_timers(Timers, MiniQMCTimerNames, timer_level_coarse);

  if (comm.root())
  {
    if (verbose)
      outputManager.setVerbosity(Verbosity::HIGH);
    else
      outputManager.setVerbosity(Verbosity::LOW);
  }

  print_version(verbose);

  SPOSet* spo_main;
  int nTiles = 1;

  ParticleSet ions;
  // initialize ions and splines which are shared by all threads later
  {
    Timers[Timer_Setup]->start();
    Tensor<OHMMS_PRECISION, 3> lattice_b;
    build_ions(ions, tmat, lattice_b);
    const int nels = count_electrons(ions, 1);
    const int norb = nels / 2;
    tileSize       = (tileSize > 0) ? tileSize : norb;
    nTiles         = norb / tileSize;

    number_of_electrons = nels;

    const size_t SPO_coeff_size =
        static_cast<size_t>(norb) * (nx + 3) * (ny + 3) * (nz + 3) * sizeof(RealType);
    const double SPO_coeff_size_MB = SPO_coeff_size * 1.0 / 1024 / 1024;

    app_summary() << "Number of orbitals/splines = " << norb << endl
                  << "Tile size = " << tileSize << endl
                  << "Number of tiles = " << nTiles << endl
                  << "Number of electrons = " << nels << endl
                  << "Rmax = " << Rmax << endl
                  << "AcceptanceRatio = " << accept << endl;
    app_summary() << "Iterations = " << nsteps << endl;
#ifdef HAVE_MPI
    app_summary() << "MPI processes = " << comm.size() << endl;
#endif
    app_summary() << "OpenMP threads = " << omp_get_max_threads() << endl;
    app_summary() << "Number of walkers per rank = " << nmovers << endl;
    app_summary() << "Number of batches = " << nbatches << endl;

    app_summary() << "\nSPO coefficients size = " << SPO_coeff_size << " bytes ("
                  << SPO_coeff_size_MB << " MB)" << endl;
    app_summary() << "delayed update rank = " << delay_rank << endl;

    spo_main = build_SPOSet(useRef, nx, ny, nz, norb, nTiles, lattice_b);
    Timers[Timer_Setup]->stop();
  }

  if (!useRef)
    app_summary() << "Using SoA distance table, Jastrow + einspline, " << endl
                  << "and determinant update." << endl << endl;
  else
    app_summary() << "Using the reference implementation for Jastrow, " << endl
                  << "determinant update, and distance table + einspline of the " << endl
                  << "reference implementation " << endl << endl;

  Timers[Timer_Total]->start();

  Timers[Timer_Init]->start();
  std::vector<Mover*> mover_list(nmovers, nullptr);

  // prepare movers
  #pragma omp parallel for
  for (int iw = 0; iw < nmovers; iw++)
  {
    // create and initialize movers
    Mover* thiswalker = new Mover(myPrimes[iw], ions, Rmax);
    mover_list[iw]    = thiswalker;

    // create wavefunction per mover
    build_WaveFunction(useRef, spo_main, thiswalker->wavefunction, ions, thiswalker->els, thiswalker->rng, delay_rank, enableJ3);

    // initial computing
    thiswalker->els.update();
  }

  // initial computing
  #pragma omp parallel for
  for(int batch = 0; batch < nbatches; batch++)
  {
    int first, last;
    FairDivideLow(mover_list.size(), nbatches, batch, first, last);
    const std::vector<ParticleSet*> P_list(extract_els_list(extract_sub_list(mover_list, first, last)));
    const std::vector<WaveFunction*> WF_list(extract_wf_list(extract_sub_list(mover_list, first, last)));
    mover_list[first]->wavefunction.flex_evaluateLog(WF_list, P_list);
  }
  Timers[Timer_Init]->stop();

  const int nions    = ions.getTotalNum();
  const int nels     = mover_list[0]->els.getTotalNum();
  const int nels3    = 3 * nels;

  // this is the number of qudrature points for the non-local PP
  const int nknots(mover_list[0]->nlpp.size());

  for (int mc = 0; mc < nsteps; ++mc)
  {
    #pragma omp parallel for
    for(int batch = 0; batch < nbatches; batch++)
    {
      Timers[Timer_Diffusion]->start();

      int first, last;
      FairDivideLow(mover_list.size(), nbatches, batch, first, last);
      const std::vector<Mover*> Sub_list(extract_sub_list(mover_list, first, last));
      const std::vector<ParticleSet*> P_list(extract_els_list(Sub_list));
      const std::vector<WaveFunction*> WF_list(extract_wf_list(Sub_list));
      const Mover& anon_mover = *Sub_list[0];

      int nw_this_batch = last - first;
      int nw_this_batch_3 = nw_this_batch * 3;

      std::vector<GradType> grad_now(nw_this_batch);
      std::vector<GradType> grad_new(nw_this_batch);
      std::vector<ValueType> ratios(nw_this_batch);
      std::vector<PosType> delta(nw_this_batch);
      aligned_vector<RealType> ur(nw_this_batch);
      /// masks for movers with valid moves
      std::vector<int> isValid(nw_this_batch);

      // synchronous walker moves
      for (int l = 0; l < nsubsteps; ++l) // drift-and-diffusion
      {
        for (int iel = 0; iel < nels; ++iel)
        {
	  // Operate on electron with index iel
          anon_mover.els.flex_setActive(P_list, iel);

          // Compute gradient at the current position
          Timers[Timer_evalGrad]->start();
          anon_mover.wavefunction.flex_evalGrad(WF_list, P_list, iel, grad_now);
          Timers[Timer_evalGrad]->stop();

          // Construct trial move
          Sub_list[0]->rng.generate_uniform(ur.data(), nw_this_batch);
          Sub_list[0]->rng.generate_normal(&delta[0][0], nw_this_batch_3);
          anon_mover.els.flex_makeMove(P_list, iel, delta);

          std::vector<bool> isAccepted(Sub_list.size());

          // Compute gradient at the trial position
          Timers[Timer_ratioGrad]->start();
          anon_mover.wavefunction.flex_ratioGrad(WF_list, P_list, iel, ratios, grad_new);
          Timers[Timer_ratioGrad]->stop();

          // Accept/reject the trial move
          for (int iw = 0; iw < nw_this_batch; iw++)
            if (ur[iw] < accept)
              isAccepted[iw] = true;
            else
              isAccepted[iw] = false;

          Timers[Timer_Update]->start();
          // update WF storage
          anon_mover.wavefunction.flex_acceptrestoreMove(WF_list, P_list, isAccepted, iel);
          Timers[Timer_Update]->stop();

          // Update position
          for (int iw = 0; iw < nw_this_batch; iw++)
          {
            if (isAccepted[iw]) // MC
              Sub_list[iw]->els.acceptMove(iel);
            else
              Sub_list[iw]->els.rejectMove(iel);
          }
        } // iel
      } // substeps

      Timers[Timer_Update]->start();
      for (int iw = 0; iw < nw_this_batch; iw++)
        WF_list[iw]->completeUpdates();
      Timers[Timer_Update]->stop();

      for (int iw = 0; iw < nw_this_batch; iw++)
        Sub_list[iw]->els.donePbyP();

      // evaluate Kinetic Energy
      anon_mover.wavefunction.flex_evaluateGL(WF_list, P_list);

      Timers[Timer_Diffusion]->stop();

      if(!run_pseudo) continue;

      // Compute NLPP energy using integral over spherical points
      // Ye: I have not found a strategy for NLPP
      Timers[Timer_ECP]->start();
      #pragma omp parallel for
      for (int iw = first; iw < last; iw++)
      {
        mover_list[iw]->nlpp.evaluate(mover_list[iw]->els, mover_list[iw]->wavefunction);
      }
      Timers[Timer_ECP]->stop();
    } // batch
  } // nsteps
  Timers[Timer_Total]->stop();

  // free all movers
  #pragma omp parallel for
  for (int iw = 0; iw < nmovers; iw++)
    delete mover_list[iw];
  mover_list.clear();
  delete spo_main;

  if (comm.root())
  {
    cout << "================================== " << endl;

    TimerManager.print();

    cout << endl << "========== Throughput ============ " << endl << endl;
    cout << "Total throughput ( N_walkers * N_elec^3 / Total time ) = "
         << (nmovers * comm.size() * std::pow(double(nels),3) / Timers[Timer_Total]->get_total()) << std::endl;
    cout << "Diffusion throughput ( N_walkers * N_elec^3 / Diffusion time ) = "
         << (nmovers * comm.size() * std::pow(double(nels),3) / Timers[Timer_Diffusion]->get_total()) << std::endl;
    if(run_pseudo)
      cout << "Pseudopotential throughput ( N_walkers * N_elec^2 / Pseudopotential time ) = "
           << (nmovers * comm.size() * std::pow(double(nels),2) / Timers[Timer_ECP]->get_total()) << std::endl;
    cout << endl;

    XMLDocument doc;
    XMLNode* resources = doc.NewElement("resources");
    XMLNode* hardware  = doc.NewElement("hardware");
    resources->InsertEndChild(hardware);
    doc.InsertEndChild(resources);
    XMLNode* timing = TimerManager.output_timing(doc);
    resources->InsertEndChild(timing);

    XMLNode* particle_info = doc.NewElement("particles");
    resources->InsertEndChild(particle_info);
    XMLNode* electron_info = doc.NewElement("particle");
    electron_info->InsertEndChild(MakeTextElement(doc, "name", "e"));
    electron_info->InsertEndChild(MakeTextElement(doc, "size", std::to_string(number_of_electrons)));
    particle_info->InsertEndChild(electron_info);


    XMLNode* run_info    = doc.NewElement("run");
    XMLNode* driver_info = doc.NewElement("driver");
    driver_info->InsertEndChild(MakeTextElement(doc, "name", "miniqmc"));
    driver_info->InsertEndChild(MakeTextElement(doc, "steps", std::to_string(nsteps)));
    driver_info->InsertEndChild(MakeTextElement(doc, "substeps", std::to_string(nsubsteps)));
    run_info->InsertEndChild(driver_info);
    resources->InsertEndChild(run_info);

    std::string info_name =
        "info_" + std::to_string(na) + "_" + std::to_string(nb) + "_" + std::to_string(nc) + ".xml";
    doc.SaveFile(info_name.c_str());
  }

  return 0;
}
