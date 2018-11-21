#include "Drivers/MiniqmcDriverFunctions.hpp"

namespace qmcplusplus
{
template<>
void MiniqmcDriverFunctions<Devices::KOKKOS>::initialize(int argc, char** argv)
{
  Kokkos::initialize(argc, argv);
}

template<Devices DT>
void MiniqmcDriverFunctions<DT>::initialize(int argc, char** argv)
{}


template<Devices DT>
void MiniqmcDriverFunctions<DT>::updateFromDevice(
    DiracDeterminant<DeterminantDeviceImp<DT>>& determinant_device)
{}

template<Devices DT>
void MiniqmcDriverFunctions<DT>::thread_main(const int ip,
					     const int team_size,
					     MiniqmcOptions& mq_opt,
					     const PrimeNumberSet<uint32_t>& myPrimes,
					     ParticleSet ions,
					     const SPOSet* spo_main)
{
  const int nels  = count_electrons(ions,1);
  const int nels3 = 3 * nels;

  const int nions = ions.getTotalNum();
  //const int nels  = mover_list[0]->els.getTotalNum();
  // this is the number of quadrature points for the non-local PP
  // Clearly this is not general, but for now, 12 point quadrature is hard coded in 
  // NonLocalPP.  Thus, we bypass the need to initialize the whole set of movers to 
  // read this hard coded number.
  constexpr int nknots = 12; 

  // For VMC, tau is large and should result in an acceptance ratio of roughly
  // 50%
  // For DMC, tau is small and should result in an acceptance ratio of 99%
  constexpr QMCT::RealType tau = 2.0;

  QMCT::RealType sqrttau = std::sqrt(tau);
  QMCT::RealType accept  = 0.5;

  //MiniqmcDriverFunctions<DT>::mover_info();
  std::cout << "Thread/Partition ID:" << ip << '\n';
  //Since we've merged initialization and execution, we get rid of the
  // mover_list vector.
  const int teamID = ip;
  mq_opt.Timers[mq_opt.Timer_Init]->start();
  // create and initialize movers
  Mover thiswalker(myPrimes[teamID], ions);
  // create a spo view in each Mover
  thiswalker.spo = build_SPOSet_view(mq_opt.useRef, spo_main, team_size, teamID);

  // create wavefunction per mover
  // This updates ions
  // build_WaveFunction is not thread safe!
  build_WaveFunction(mq_opt.useRef, thiswalker.wavefunction, ions, thiswalker.els, thiswalker.rng, mq_opt.enableJ3);

  // initial computing
  thiswalker.els.update();
  thiswalker.wavefunction.evaluateLog(thiswalker.els);
  mq_opt.Timers[mq_opt.Timer_Init]->stop();

  auto& els          = thiswalker.els;
  auto& spo          = *thiswalker.spo;
  auto& random_th    = thiswalker.rng;
  auto& wavefunction = thiswalker.wavefunction;
  auto& ecp          = thiswalker.nlpp;

  ParticleSet::ParticlePos_t delta(nels);
  ParticleSet::ParticlePos_t rOnSphere(nknots);

  aligned_vector<QMCT::RealType> ur(nels);

  int my_accepted = 0;
  for (int mc = 0; mc < mq_opt.nsteps; ++mc)
  {
    mq_opt.Timers[mq_opt.Timer_Diffusion]->start();
    for (int l = 0; l < mq_opt.nsubsteps; ++l) // drift-and-diffusion
    {
      random_th.generate_uniform(ur.data(), nels);
      random_th.generate_normal(&delta[0][0], nels3);
      for (int iel = 0; iel < nels; ++iel)
      {
        // Operate on electron with index iel
        els.setActive(iel);
        // Compute gradient at the current position
        mq_opt.Timers[mq_opt.Timer_evalGrad]->start();
	ParticleSet::PosType grad_now = wavefunction.evalGrad(els, iel);
        mq_opt.Timers[mq_opt.Timer_evalGrad]->stop();

        // Construct trial move
	ParticleSet::PosType dr   = sqrttau * delta[iel];
        bool isValid = els.makeMoveAndCheck(iel, dr);

        if (!isValid)
          continue;

        // Compute gradient at the trial position
        mq_opt.Timers[mq_opt.Timer_ratioGrad]->start();

	ParticleSet::PosType grad_new;
        wavefunction.ratioGrad(els, iel, grad_new);

        spo.evaluate_vgh(els.R[iel]);

        mq_opt.Timers[mq_opt.Timer_ratioGrad]->stop();

        // Accept/reject the trial move
        if (ur[iel] > accept) // MC
        {
          // Update position, and update temporary storage
          mq_opt.Timers[mq_opt.Timer_Update]->start();
          wavefunction.acceptMove(els, iel);
          mq_opt.Timers[mq_opt.Timer_Update]->stop();
          els.acceptMove(iel);
          my_accepted++;
        }
        else
        {
          els.rejectMove(iel);
          wavefunction.restore(iel);
        }
      } // iel
    }   // substeps

    els.donePbyP();

    // evaluate Kinetic Energy
    wavefunction.evaluateGL(els);

    mq_opt.Timers[mq_opt.Timer_Diffusion]->stop();

    // Compute NLPP energy using integral over spherical points

    ecp.randomize(rOnSphere); // pick random sphere
    const DistanceTableData* d_ie = els.DistTables[wavefunction.get_ei_TableID()];

    mq_opt.Timers[mq_opt.Timer_ECP]->start();
    for (int jel = 0; jel < els.getTotalNum(); ++jel)
    {
      const auto& dist  = d_ie->Distances[jel];
      const auto& displ = d_ie->Displacements[jel];
      for (int iat = 0; iat < nions; ++iat)
        if (dist[iat] < mq_opt.Rmax)
          for (int k = 0; k < nknots; k++)
          {
	    ParticleSet::PosType deltar(dist[iat] * rOnSphere[k] - displ[iat]);

            els.makeMoveOnSphere(jel, deltar);

            mq_opt.Timers[mq_opt.Timer_Value]->start();
            spo.evaluate_v(els.R[jel]);
            wavefunction.ratio(els, jel);
            mq_opt.Timers[mq_opt.Timer_Value]->stop();

            els.rejectMove(jel);
          }
    }
    mq_opt.Timers[mq_opt.Timer_ECP]->stop();

  } // nsteps
}

template<Devices DT>
void MiniqmcDriverFunctions<DT>::runThreads(MiniqmcOptions& mq_opt,
                                            const PrimeNumberSet<uint32_t>& myPrimes,
                                            ParticleSet& ions,
					    const SPOSet* spo_main)
{
  MiniqmcDriverFunctions<DT>::thread_main(1, 1, mq_opt, myPrimes, ions, spo_main);
}

template<>
void MiniqmcDriverFunctions<Devices::KOKKOS>::runThreads(MiniqmcOptions& mq_opt,
                                                         const PrimeNumberSet<uint32_t>& myPrimes,
                                                         ParticleSet& ions,
							 const SPOSet* spo_main)
{
  auto main_function = KOKKOS_LAMBDA(int thread_id, int team_size)
  {
    printf(" thread_id = %d\n", thread_id);
    MiniqmcDriverFunctions<Devices::KOKKOS>::thread_main(thread_id,
							 team_size,
							 const_cast<MiniqmcOptions&>(mq_opt),
                                                         myPrimes,
                                                         ions,
							 spo_main);
  };
  #if defined(KOKKOS_ENABLE_OPENMP) && !defined(KOKKOS_ENABLE_CUDA)
  int num_threads = Kokkos::OpenMP::thread_pool_size();

  int crewsize = std::max(1, num_threads / mq_opt.ncrews);
  printf(" In partition master with %d threads, %d crews, and %d movers.  Crewsize = %d \n",
         num_threads,
         mq_opt.ncrews,
         mq_opt.nmovers,
         crewsize);
  Kokkos::OpenMP::partition_master(main_function, mq_opt.nmovers, crewsize);
#else
  main_function(0, 1);
#endif
}

  template class MiniqmcDriverFunctions<Devices::KOKKOS>;
  template class MiniqmcDriverFunctions<Devices::CPU>;
}

