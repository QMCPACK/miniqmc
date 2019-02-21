#include "Drivers/MiniqmcDriverFunctions.hpp"
#include "Drivers/Movers.hpp"
namespace qmcplusplus
{
template<Devices DT>
void MiniqmcDriverFunctions<DT>::initialize(int argc, char** argv)
{}

template<Devices DT>
void MiniqmcDriverFunctions<DT>::updateFromDevice(DiracDeterminant<DeterminantDeviceImp<DT>>& determinant_device)
{}

template<Devices DT>
void MiniqmcDriverFunctions<DT>::thread_main(const int ip,
                                                       const int team_size,
                                                       MiniqmcOptions& mq_opt,
                                                       const PrimeNumberSet<uint32_t>& myPrimes,
                                                       ParticleSet ions,
                                                       const SPOSet* spo_main)
{
  const int member_id = ip % team_size;
  // create and initialize movers
  app_summary() << "thread:" << ip << " starting up \n";

  Mover* thiswalker = new Mover(myPrimes[ip], ions);
  //mover_list[iw]    = thiswalker;

  // create a spo view in each Mover
  thiswalker->spo = SPOSetBuilder<Devices::CPU>::buildView(mq_opt.useRef, spo_main, team_size, member_id);

  // create wavefunction per mover
  WaveFunctionBuilder<Devices::CPU>::build(mq_opt.useRef,
                                           thiswalker->wavefunction,
                                           ions,
                                           thiswalker->els,
                                           thiswalker->rng,
                                           mq_opt.enableJ3);

  // initial computing
  thiswalker->els.update();

  thiswalker->wavefunction.evaluateLog(thiswalker->els);


  const int nions = ions.getTotalNum();
  const int nels  = thiswalker->els.getTotalNum();
  const int nels3 = 3 * nels;

  // this is the number of quadrature points for the non-local PP
  const int nknots(thiswalker->nlpp.size());

  // For VMC, tau is large and should result in an acceptance ratio of roughly
  // 50%
  // For DMC, tau is small and should result in an acceptance ratio of 99%
  const QMCT::RealType tau = 2.0;

  QMCT::RealType sqrttau = std::sqrt(tau);

  auto& els          = thiswalker->els;
  auto& spo          = *thiswalker->spo;
  auto& random_th    = thiswalker->rng;
  auto& wavefunction = thiswalker->wavefunction;
  auto& ecp          = thiswalker->nlpp;

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
        QMCT::PosType grad_now = wavefunction.evalGrad(els, iel);
        mq_opt.Timers[Timer_evalGrad]->stop();

        // Construct trial move
        QMCT::PosType dr = sqrttau * delta[iel];
        bool isValid     = els.makeMoveAndCheck(iel, dr);

        if (!isValid)
          continue;

        // Compute gradient at the trial position
        mq_opt.Timers[Timer_ratioGrad]->start();

        QMCT::PosType grad_new;
        wavefunction.ratioGrad(els, iel, grad_new);

        spo.evaluate_vgh(els.R[iel]);

        mq_opt.Timers[Timer_ratioGrad]->stop();

        // Accept/reject the trial move
        if (ur[iel] < mq_opt.accept) // MC
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
            QMCT::PosType deltar(dist[iat] * rOnSphere[k] - displ[iat]);

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

template<Devices DT>
void MiniqmcDriverFunctions<DT>::runThreads(MiniqmcOptions& mq_opt,
                                            const PrimeNumberSet<uint32_t>& myPrimes,
                                            ParticleSet& ions,
                                            const SPOSet* spo_main)
{
#pragma omp parallel for
  for (int iw = 0; iw < mq_opt.nmovers; iw++)
  {
    MiniqmcDriverFunctions<DT>::thread_main(iw, 1, mq_opt, myPrimes, ions, spo_main);
  }
}

template<Devices DT>
void MiniqmcDriverFunctions<DT>::movers_runThreads(MiniqmcOptions& mq_opt,
                                            const PrimeNumberSet<uint32_t>& myPrimes,
                                            ParticleSet& ions,
                                            const SPOSet* spo_main)
{
#pragma omp parallel for
  for (int iw = 0; iw < mq_opt.nmovers; iw++)
  {
    MiniqmcDriverFunctions<DT>::movers_thread_main(iw, 1, mq_opt, myPrimes, ions, spo_main);
  }
}

template class MiniqmcDriverFunctions<Devices::CPU>;
#ifdef QMC_USE_CUDA
template class MiniqmcDriverFunctions<Devices::CUDA>;
#endif
} // namespace qmcplusplus
