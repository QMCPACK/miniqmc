////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2019 QMCPACK developers.
//
// File developed by:
// Peter Doak, doakpw@ornl.gov, Oak Ridge National Lab
//
// File created by:
// Peter Doak, doakpw@ornl.gov, Oak Ridge National Lab
////////////////////////////////////////////////////////////////////////////////
// -*- C++ -*-

/**
 * @file
 * @brief implementation of Kokkos specialization of MiniqmcDriverFunctions
 */

#include "Drivers/MiniqmcDriverFunctions.hpp"
#include "Numerics/Spline2/BsplineAllocatorKOKKOS.hpp"
namespace qmcplusplus
{

template<>
void MiniqmcDriverFunctions<Devices::KOKKOS>::initialize(int argc, char** argv)
{
  Kokkos::initialize(argc, argv);
}

template<>
void MiniqmcDriverFunctions<Devices::KOKKOS>::thread_main(const int ip,
                                                          const int team_size,
                                                          MiniqmcOptions& mq_opt,
                                                          const PrimeNumberSet<uint32_t>& myPrimes,
                                                          ParticleSet ions,
                                                          const SPOSet* spo_main)
{
  constexpr Devices DT = Devices::KOKKOS;
  const int nels       = count_electrons(ions, 1);
  const int nels3      = 3 * nels;

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

  std::cout << "Thread/Partition ID:" << ip << '\n';
  //Since we've merged initialization and execution, we get rid of the
  // mover_list vector.
  const int teamID = ip;
  // create and initialize movers
  Mover thiswalker(myPrimes[teamID], ions);
  // create a spo view in each Mover
  thiswalker.spo = SPOSetBuilder<DT>::buildView(mq_opt.useRef, spo_main, team_size, teamID);

  DeviceBuffers<DT> device_buffers;

  // create wavefunction per mover
  // This updates ions
  // build_WaveFunction is not thread safe!
  WaveFunctionBuilder<DT>::build(mq_opt.useRef,
                     thiswalker.wavefunction,
                     ions,
                     thiswalker.els,
                     thiswalker.rng,
				 device_buffers,
                     mq_opt.enableJ3);

  // initial computing
  thiswalker.els.update();
  thiswalker.wavefunction.evaluateLog(thiswalker.els);
  
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
    mq_opt.Timers[Timer_Diffusion]->start();
    for (int l = 0; l < mq_opt.nsubsteps; ++l) // drift-and-diffusion
    {
      random_th.generate_uniform(ur.data(), nels);
      random_th.generate_normal(&delta[0][0], nels3);
      for (int iel = 0; iel < nels; ++iel)
      {
        // Operate on electron with index iel
        els.setActive(iel);
        // Compute gradient at the current position
        mq_opt.Timers[Timer_evalGrad]->start();
        ParticleSet::PosType grad_now = wavefunction.evalGrad(els, iel);
        mq_opt.Timers[Timer_evalGrad]->stop();

        // Construct trial move
        ParticleSet::PosType dr = sqrttau * delta[iel];
        bool isValid            = els.makeMoveAndCheck(iel, dr);

        if (!isValid)
          continue;

        // Compute gradient at the trial position
        mq_opt.Timers[Timer_ratioGrad]->start();

        ParticleSet::PosType grad_new;
        wavefunction.ratioGrad(els, iel, grad_new);

        spo.evaluate_vgh(els.R[iel]);

        mq_opt.Timers[Timer_ratioGrad]->stop();

        // Accept/reject the trial move
        if (ur[iel] > accept) // MC
        {
          // Update position, and update temporary storage
          mq_opt.Timers[Timer_Update]->start();
          wavefunction.acceptMove(els, iel);
          mq_opt.Timers[Timer_Update]->stop();
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

    mq_opt.Timers[Timer_Diffusion]->stop();

    // Compute NLPP energy using integral over spherical points

    ecp.randomize(rOnSphere); // pick random sphere
    const DistanceTableData* d_ie = els.DistTables[wavefunction.get_ei_TableID()];

    mq_opt.Timers[Timer_ECP]->start();
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

            mq_opt.Timers[Timer_Value]->start();
            spo.evaluate_v(els.R[jel]);
            wavefunction.ratio(els, jel);
            mq_opt.Timers[Timer_Value]->stop();

            els.rejectMove(jel);
          }
    }
    mq_opt.Timers[Timer_ECP]->stop();

  } // nsteps
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


template<>
void MiniqmcDriverFunctions<Devices::KOKKOS>::movers_runThreads(MiniqmcOptions& mq_opt,
                                            const PrimeNumberSet<uint32_t>& myPrimes,
                                            ParticleSet& ions,
                                            const SPOSet* spo_main)
{
  auto main_function = KOKKOS_LAMBDA(int thread_id, int team_size)
  {
    printf(" thread_id = %d\n", thread_id);
    TaskBlockBarrier<Threading::KOKKOS> barrier(team_size); /// maybe

    MiniqmcDriverFunctions<Devices::KOKKOS>::crowd_thread_main(thread_id, 							       barrier,
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
  main_function(0,1);
  #endif

}

template class MiniqmcDriverFunctions<Devices::KOKKOS>;

} // namespace qmcplusplus
