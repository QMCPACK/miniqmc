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
  app_summary() << "  miniqmc   [-bhjvV] [-g \"n0 n1 n2\"] [-m meshfactor]"      << '\n';
  app_summary() << "            [-n steps] [-N substeps] [-x rmax]"              << '\n';
  app_summary() << "            [-r AcceptanceRatio] [-s seed] [-w walkers]"     << '\n';
  app_summary() << "            [-a tile_size] [-t timer_level]"                 << '\n';
  app_summary() << "options:"                                                    << '\n';
  app_summary() << "  -a  size of each spline tile       default: num of orbs"   << '\n';
  app_summary() << "  -b  use reference implementations  default: off"           << '\n';
  app_summary() << "  -g  set the 3D tiling.             default: 1 1 1"         << '\n';
  app_summary() << "  -h  print help and exit"                                   << '\n';
  app_summary() << "  -j  enable three body Jastrow      default: off"           << '\n';
  app_summary() << "  -m  meshfactor                     default: 1.0"           << '\n';
  app_summary() << "  -n  number of MC steps             default: 5"             << '\n';
  app_summary() << "  -N  number of MC substeps          default: 1"             << '\n';
  app_summary() << "  -r  set the acceptance ratio.      default: 0.5"           << '\n';
  app_summary() << "  -s  set the random seed.           default: 11"            << '\n';
  app_summary() << "  -t  timer level: coarse or fine    default: fine"          << '\n';
  app_summary() << "  -w  number of walker(movers)       default: num of threads"<< '\n';
  app_summary() << "  -v  verbose output"                                        << '\n';
  app_summary() << "  -V  print version information and exit"                    << '\n';
  app_summary() << "  -x  set the Rmax.                  default: 1.7"           << '\n';
  // clang-format on
}

int main(int argc, char** argv)
{
  Kokkos::initialize(argc, argv);
  {
    // clang-format off
    typedef QMCTraits::RealType           RealType;
    typedef ParticleSet::ParticlePos_t    ParticlePos_t;
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
    int nmovers = 1; // LNS -- changed from number of threads.  Must use w option
    // thread blocking
    int tileSize  = -1;
    int team_size = 1;
    int nsubsteps = 1;
    // Set cutoff for NLPP use.
    RealType Rmax(1.7);
    RealType accept  = 0.5;
    bool useRef   = false;
    bool enableJ3 = false;
    
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
      if ((opt = getopt(argc, argv, "bhjvVa:c:g:m:n:N:r:s:t:w:x:")) != -1)
      {
	switch (opt)
	{
	case 'a':
	  tileSize = atoi(optarg);
	  break;
	case 'b':
	  useRef = true;
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
	case 'r':
	  accept = atof(optarg);
	  break;
	case 's':
	  iseed = atoi(optarg);
	  break;
	case 't':
	  timer_level_name = std::string(optarg);
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

    // use this pointer if I'm doing reference, but acutal spo if not
    SPOSet* spo_main;
    using spo_type = einspline_spo<OHMMS_PRECISION, 32>;
    using mover_type = mover<OHMMS_PRECISION, 32>;
    spo_type spo;

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
      //    app_summary() << "OpenMP threads = " << omp_get_max_threads() << endl;
#ifdef HAVE_MPI
      app_summary() << "MPI processes = " << comm.size() << endl;
#endif
      
      app_summary() << "\nSPO coefficients size = " << SPO_coeff_size << " bytes ("
		    << SPO_coeff_size_MB << " MB)" << endl;

      if (useRef) {
	spo_main = build_SPOSet(useRef, nx, ny, nz, norb, nTiles, lattice_b);
      } else {
	spo.set(nx, ny, nz, norb);
	spo.Lattice.set(lattice_b);
      }
      
      Timers[Timer_Setup]->stop();
    }

    if (!useRef)
      app_summary() << "Using SoA distance table, Jastrow + einspline, " << endl
		    << "and determinant update." << endl;
    else
      app_summary() << "Using the reference implementation for Jastrow, " << endl
		    << "determinant update, and distance table + einspline of the " << endl
		    << "reference implementation " << endl;
    
    Timers[Timer_Total]->start();
    
    Timers[Timer_Init]->start();
   
    // use this if doing ref
    std::vector<Mover_ref*> mover_list_ref(nmovers, nullptr);
    // use this if not
    std::vector<mover_type*> mover_list(nmovers, nullptr);
    
    // prepare movers
    // LNS -- something interesting is happening here
    //        I should probably make this a lambda and call it with nmovers and
    //        team_size which are assigned above
    //        The issue is that we could have a different number of movers and
    //        thread teams
    for (int iw = 0; iw < nmovers; iw++)
    {
      const int ip        = omp_get_thread_num();
      const int member_id = ip % team_size;
      
      // create and initialize movers
      if (useRef) {      
	Mover_ref* thiswalker = new Mover_ref(myPrimes[ip], ions);
	mover_list_ref[iw]    = thiswalker;
      } else {
	mover_type* thiswalker = new mover_type(myPrimes[i], ions, spo);
	mover_list[iw]    = thiswalker;
      }
	
      // create a spo view in each Mover
      if (useRef) {
	mover_list_ref[iw]->spo = build_SPOSet_view(useRef, spo_main, team_size, member_id);
      }

      // create wavefunction per mover
      if (useRef) {
	build_WaveFunction(useRef, mover_list_ref[iw]->wavefunction, ions, mover_list_ref[iw]->els, mover_list_ref[iw]->rng, enableJ3);
      } else {
	build_WaveFunction(useRef, mover_list[iw]->wavefunction, ions, mover_list[iw]->els, mover_list[iw]->rng, enableJ3);
      }
      
      // initial computing
      if (useRef) {
	mover_list_ref[iw]->els.update(); // basically goes through the distance tables and calls evaluate from the current particleset
      } else {
	mover_list[iw]->els.update();
      }
      
      // LNS new -- push necessary contents of els particleset up to the device
      //            note that by now, the electron-ion distance table is present in els
      if (useRef == false)
      {
	mover_list[iw]->els.PushDataToParticleSetKokkos();
      }
    }
    
    { // initial computing
      if (useRef) {
	const std::vector<ParticleSet*> P_list(extract_els_list(mover_list_ref));
	const std::vector<WaveFunction*> WF_list(extract_wf_list(mover_list_ref));
	mover_list_ref[0]->wavefunction.multi_evaluateLog(WF_list, P_list);
      } else {
	const std::vector<ParticleSet*> P_list(extract_els_list(mover_list));
	const std::vector<WaveFunction*> WF_list(extract_wf_list(mover_list));
	mover_list[0]->wavefunction.multi_evaluateLog(WF_list, P_list);
      }
    }
    Timers[Timer_Init]->stop();
    
    const int nions    = ions.getTotalNum();
    const int nels     = mover_list[0]->els.getTotalNum();
    const int nels3    = 3 * nels;
    const int nmovers3 = 3 * nmovers;
    
    // this is the number of qudrature points for the non-local PP
    int nknots;
    if (useRef) {
      nknots = mover_list_ref[0]->nlpp.size();
    } else {
      nknots = mover_list[0]->nlpp.size();
    }
    
    // For VMC, tau is large and should result in an acceptance ratio of roughly
    // 50%
    // For DMC, tau is small and should result in an acceptance ratio of 99%
    const RealType tau = 2.0;
    
    RealType sqrttau = std::sqrt(tau);
    
    // synchronous walker moves
    {
      std::vector<PosType> delta(nmovers);
      std::vector<PosType> pos_list(nmovers);
      std::vector<GradType> grad_now(nmovers);
      std::vector<GradType> grad_new(nmovers);
      std::vector<ValueType> ratios(nmovers);
      aligned_vector<RealType> ur(nmovers);
      /// masks for movers with valid moves
      std::vector<int> isValid(nmovers);

      // LNS: Note that we are assembling lists of vector<ParticleSet*> or other 
      //      things like it way too often because interoperability with the 
      //      reference version discourages making data whose types don't make
      //      sense to the reference.  Could come up with a design that avoids
      //      this and saves a lot of pointless deep copying of pointers to the device
      //      If you did this, would also likely re-plumb the wavefunction_multi method
      //      so that perhaps things like J1 and J2 could avoid this as well
      for (int mc = 0; mc < nsteps; ++mc)
      {
	Timers[Timer_Diffusion]->start();

	std::vector<ParticleSet*> P_list;
	std::vector<WaveFunction*> WF_list;
	if (useRef) {
	  P_list(extract_els_list(mover_list_ref));
	  WF_list(extract_wf_list(mover_list_ref));
	} else {
	  P_list = extract_els_list(mover_list);
	  WF_list = extract_wf_list(mover_list);
	}

	Mover* anon_mover_ref;
	mover_type* anon_mover;
	if (useRef) {
	  anon_mover_ref = mover_list_ref[0];
	} else {
	  anon_mover = mover_list[0];
	}
	  
	for (int l = 0; l < nsubsteps; ++l) // drift-and-diffusion
	{
	  for (int iel = 0; iel < nels; ++iel)
	  {
	    // Operate on electron with index iel
	    if (useRef) {
	      for (int iw = 0; iw < nmovers; iw++)
		mover_list_ref[iw]->els.setActive(iel); // watch out for this, does a bunch of evaluates on distanceTables
	    } else {
	      anon_mover->els.multi_setActiveKokkos(P_list,iel); // note this is only doing this on the device, not the host
	    }
	    // Compute gradient at the current position
	    Timers[Timer_evalGrad]->start();
	    if (useRef) {
	      anon_mover_ref->wavefunction.multi_evalGrad(WF_list, P_list, iel, grad_now);
	    } else {
	      anon_mover->wavefunction.multi_evalGrad(WF_list, P_list, iel, grad_now);
	    }
	    Timers[Timer_evalGrad]->stop();
	    
	    // Construct trial move
	    if (useRef) {
	      mover_list_ref[0]->rng.generate_uniform(ur.data(), nmovers);
	      mover_list_ref[0]->rng.generate_normal(&delta[0][0], nmovers3);
	    } else {
	      mover_list[0]->rng.generate_uniform(ur.data(), nmovers);
	      mover_list[0]->rng.generate_normal(&delta[0][0], nmovers3);
	    }
	      
	    if (useRef) {
	      for (int iw = 0; iw < nmovers; iw++) {
		  PosType dr  = sqrttau * delta[iw];
		  isValid[iw] = mover_list_ref[iw]->els.makeMoveAndCheck(iel, dr);
		}
	    } else {
	      Kokkos::View<RealType*[3]> dr("dr", nmovers);
	      auto drMirror = Kokkos::create_mirror_view(dr);
	      for (int iw = 0; iw < nmovers; iw++) {
		for (int d = 0; d < 3; d++) {
		  drMirror(iw,d) = sqrttau * delta[iw][d];
		}
	      }
	      Kokkos::deep_copy(dr, drMirror);
	      anon_mover->els.multi_makeMoveAndCheckKokkos(P_list, dr, iel, isValid);
	    }

	    std::vector<ParticleSet*> valid_P_list;
	    std::vector<SPOSet*> valid_spo_list;
	    std::vector<WaveFunction*> valid_WF_list;
	    std::vector<bool> isAccepted;	    
	    if (useRef) {
	      const std::vector<Mover*> valid_mover_list(filtered_list(mover_list_ref, isValid));
	      isAccepted.resize(valid_mover_list.size());
	      valid_P_list = extract_els_list(valid_mover_list);
	      valid_spo_list = extract_spo_list(valid_mover_list);
	      valid_WF_list = extract_wf_list(valid_mover_list);
	    } else {
	      const std::vector<mover_type*> valid_mover_list(filtered_list(mover_list, isValid));
	      isAccepted.resize(valid_mover_list.size());
	      valid_P_list = extract_els_list(valid_mover_list);
	      valid_spo_list = extract_spo_list(valid_mover_list);
	      valid_WF_list = extract_wf_list(valid_mover_list);
	    }
	    
	    // Compute gradient at the trial position
	    Timers[Timer_ratioGrad]->start();
	    if (useRef) {
	      anon_mover_ref->wavefunction.multi_ratioGrad(valid_WF_list, valid_P_list, iel, ratios, grad_new);
	    } else {
	      anon_mover->wavefunction.multi_ratioGrad(valid_WF_list, valid_P_list, iel, ratios, grad_new);
	    }

	    if (useRef) {
	      for (int iw = 0; iw < valid_P_list.size(); iw++)
		pos_list[iw] = valid_P_list[iw].R[iel];
	      anon_mover_ref.spo->multi_evaluate_vgh(valid_spo_list, pos_list); 
	    } else {
	      Kokkos::View<double*[3],Kokkos::LayoutLeft> pos_list("positions", valid_P_list.size());
	      Kokkos::View<valid_P_list[0]::pskType*> allParticleSetData("apsd", valid_P_list.size());
	      auto apsdMirror = Kokkos::create_mirror_view(allParticleSetData);
	      for (int i = 0; i < valid_P_list.size(); i++) {
		apsdMirror(i) = valid_P_list[i]->psk;
	      }
	      Kokkos::deep_copy(allParticleSetData, apsdMirror);
	      Kokkos::parallel_for("populatePositions", valid_P_list.size(),
				   KOKKOS_LAMBDA(const int& i) {
				     double x, y, z;
				     
				     allParticleSetData(i).toUnit_floor(allParticleSetData(i).R(iel,0),
									allParticleSetData(i).R(iel,1),
									allParticleSetData(i).R(iel,2),
									x,y,z);
				     pos_list(i,0) = x;
				     pos_list(i,1) = y;
				     pos_list(i,2) = z;
				   });
	      const std::vector<mover_type*> valid_mover_list(filtered_list(mover_list, isValid));
	      auto vals = extract_spo_psi_list(valid_mover_list);
	      auto grads = extract_spo_grad_list(valid_mover_list);
	      auto hesss = extract_spo_grad_list(valid_mover_list);
	      spo.multi_evaluate_vgh(pos_list, vals, grads, hesss);
	    }
	    Timers[Timer_ratioGrad]->stop();
	    
	    // Accept/reject the trial move
	    for (int iw = 0; iw < isAccepted.size(); iw++)
	      if (ur[iw] < accept)
		isAccepted[iw] = true;
	      else
		isAccepted[iw] = false;
	    
	    Timers[Timer_Update]->start();
	    // update WF storage
	    if (useRef) {
	      anon_mover_ref->wavefunction.multi_acceptrestoreMove(valid_WF_list, valid_P_list, isAccepted, iel);
	    } else {
	      anon_mover->wavefunction.multi_acceptrestoreMove(valid_WF_list, valid_P_list, isAccepted, iel);
	    }
	    Timers[Timer_Update]->stop();
	    
	    // Update position
	    if (useRef) {
	      const std::vector<Mover*> valid_mover_list(filtered_list(mover_list_ref, isValid));
	      for (int iw = 0; iw < valid_mover_list.size(); iw++)
		{
		  if (isAccepted[iw]) // MC
		    valid_mover_list[iw]->els.acceptMove(iel);  // lots of calls to distanceTables.update
		  else
		    valid_mover_list[iw]->els.rejectMove(iel);  // just sets activePtcl to -1
		}
	    } else {
	      anon_mover->els.multi_acceptRejectMoveKokkos(valid_P_list, isAccepted, iel);
	    }
	  } // iel
	}   // substeps
	
	// LNS -- need to get rid of openmp
	if (useRef) {
	  for (int iw = 0; iw < nmovers; iw++)
	    {
	      mover_list[iw]->els.donePbyP();   // just sets activePtcl to -1
	      // evaluate Kinetic Energy
	    }
	} else {
	  anon_mover->els.multi_DonePbyP(P_list);
	}
	  
        if (useRef) {
	  anon_mover_ref->wavefunction.multi_evaluateGL(WF_list, P_list);   // look at what is needed here
	} else {
	  anon_mover->wavefunction.multi_evaluateGL(WF_list, P_list);   // look at what is needed here
	}
	Timers[Timer_Diffusion]->stop();
	
	// Compute NLPP energy using integral over spherical points
	// Ye: I have not found a strategy for NLPP
	Timers[Timer_ECP]->start();
	if (useRef) {
	  for (int iw = 0; iw < nmovers; iw++)
	  {
	    auto& els          = mover_list[iw]->els;
	    auto& spo          = *mover_list[iw]->spo;
	    auto& wavefunction = mover_list[iw]->wavefunction;
	    auto& ecp          = mover_list[iw]->nlpp;
	    
	    ParticlePos_t rOnSphere(nknots);
	    ecp.randomize(rOnSphere); // pick random sphere
	    const DistanceTableData* d_ie = els.DistTables[wavefunction.get_ei_TableID()];
	    
	    for (int jel = 0; jel < els.getTotalNum(); ++jel)
	    {
	      const auto& dist  = d_ie->Distances[jel];   // get distances for this electron
	      const auto& displ = d_ie->Displacements[jel];  // get displacements for this electron
	      for (int iat = 0; iat < nions; ++iat)
		if (dist[iat] < Rmax)   // check whether the distance to this ion is less than a cutoff
		  for (int k = 0; k < nknots; k++)
		  {
		    PosType deltar(dist[iat] * rOnSphere[k] - displ[iat]);
		    
		    els.makeMoveOnSphere(jel, deltar);  // sets activePtcl and calls DistTables->moveOnSphere
		    // that does update some stuff, but can probably just figure out what is needed to get R for jastrow ratios and
		    // make sure that it ends up in a kernel
		    Timers[Timer_Value]->start();
		    spo.evaluate_v(els.activePos);  // call shouldn't depend on other calls, just positions
		    wavefunction.ratio(els, jel);  // note that the particleset isn't even used here for the spo part
		    // it does come in for the jastrows though
		    Timers[Timer_Value]->stop();
		    
		    els.rejectMove(jel);  // note just sets activePtcl back to -1
		  }
	    }
	  }
	} else { // not the reference implementation
	  // strategy.  
	  // 1. Have a loop that makes a list for each walker of all electron-ion pairs that are active
	  //    a. hope that the number of such pairs is basically equal for all walkers
	  // 2. make a parallel loop where for every walker we are doing all knots for a given e-i pair at once
	  //    a. will need to separate out data so that we are handing activeParticle, activePos, temp_r and temp_dr
          //       to the wavefunction components.  Also see what is needed for a general spo.evaluate_v that is not
          //       just one move per walker
          // 3. Could eventually think about having a general batch size where it was not just one e-i pair per walker
	}    


	Timers[Timer_ECP]->stop();
      } // nsteps
    }
    Timers[Timer_Total]->stop();
    
    // free all movers
    for (int iw = 0; iw < nmovers; iw++)
      delete mover_list[iw];
    mover_list.clear();
    delete spo_main;
    
    if (comm.root())
    {
      cout << "================================== " << endl;
      
      TimerManager.print();
      
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
  }
  Kokkos::finalize();
  return 0;
}
