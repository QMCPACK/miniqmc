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

#ifndef QMCPLUSPLUS_MINIQMC_DRIVER_FUNCTIONS_HPP
#define QMCPLUSPLUS_MINIQMC_DRIVER_FUNCTIONS_HPP

#include <functional>
#include <boost/hana/map.hpp>
//#include <boost/thread/barrier.hpp>
#include "Devices.h"
#include "Drivers/MiniqmcOptions.hpp"
#include "Drivers/Mover.hpp"
#include "Drivers/Crowd.hpp"
#include "Particle/ParticleSet.h"
#include "Input/nio.hpp"
#include "QMCWaveFunctions/Determinant.h"
#include "QMCWaveFunctions/DeterminantDeviceImp.h"
#include "QMCWaveFunctions/SPOSet.h"
#include "QMCWaveFunctions/SPOSet_builder.h"
#include "Utilities/TaskBlock.hpp"
#include "Utilities/qmcpack_version.h"
#include "Utilities/PrimeNumberSet.h"

#ifdef QMC_USE_CUDA
#include "Drivers/CrowdCUDA.hpp"
#endif

namespace qmcplusplus
{
/** A purely functional class implementing miniqmcdriver functions.
 *  functions can be specialized for a particular device.
 *  These are the driver steps that any device must either use the default for
 *  or specialize.
 */
template<Devices DT>
class MiniqmcDriverFunctions
{
public:
  using QMCT = QMCTraits;
  static void initialize(int arc, char** argv);

  /** build main SPOset once */
  static void buildSPOSet(SPOSet*& spo_set,
                          MiniqmcOptions& mq_opt,
                          const int norb,
                          const int nTiles,
                          const int tile_size,
                          const Tensor<OHMMS_PRECISION, 3>& lattice_b);

  /** @name Thread Launchers
   *  These launch the threads that run a QMC block. 
   *  They have no sychronization or communication during a block.
   *  No assumptions should be made that they actually execute in parallel.
   */
  ///@{

  /** This calls the old non batch threadmain, kept around for comparison.
   *  As long as single evaluates and the internal data structures in einplsine, wavefunction, etc.
   *  are retained it should work.
   */
  static void runThreads(MiniqmcOptions& mq_opt,
                         const PrimeNumberSet<uint32_t>& myPrimes,
                         ParticleSet& ions,
                         const SPOSet* spo_main);

  /** This is the new batched thread launcher.
   *  Each thread through Crowd<device> it should leave the actual method of batch evaluation to
   *  the device through specializations of crowd.
   */
  static void movers_runThreads(MiniqmcOptions& mq_opt,
                                const PrimeNumberSet<uint32_t>& myPrimes,
                                ParticleSet& ions,
                                const SPOSet* spo_main);
  ///@}

  static void finalize();

private:
  /** @name Thread Bodies
   *  Each top level thread gets one of these.
   *  They should be are not guaranteed to run concurrently.
   */
  ///@{
  template<Threading TT>
  static void crowd_thread_main(const int ip,
                                TaskBlockBarrier<TT>& barrier,
                                const int team_size,
                                MiniqmcOptions& mq_opt,
                                const PrimeNumberSet<uint32_t>& myPrimes,
                                ParticleSet ions,
                                const SPOSet* spo_main);

  /** Legacy unbatched thread body. 
   *  Depends only on "object" local data and single evaluations that use it.
   */
  static void thread_main(const int ip,
                          const int team_size,
                          MiniqmcOptions& mq_opt,
                          const PrimeNumberSet<uint32_t>& myPrimes,
                          ParticleSet ions,
                          const SPOSet* spo_main);

  ///@}

  /** Only KOKKOS needs this and I don't think this belongs in the API
   *  The Crowd<KOKKOS> should handle it's synchronization
   */
  static void updateFromDevice(DiracDeterminant<DeterminantDeviceImp<DT>>& determinant_device);
};

#ifdef QMC_USE_KOKKOS
#include "Drivers/MiniqmcDriverFunctionsKOKKOS.hpp"
#endif


/** currently used by all devices
 */
template<Devices DT>
void MiniqmcDriverFunctions<DT>::buildSPOSet(SPOSet*& spo_set,
                                             MiniqmcOptions& mq_opt,
                                             const int norb,
                                             const int nTiles,
                                             const int tile_size,
                                             const Tensor<OHMMS_PRECISION, 3>& lattice_b)
{
  spo_set =
      SPOSetBuilder<DT>::build(mq_opt.useRef, mq_opt.nx, mq_opt.ny, mq_opt.nz, norb, nTiles, tile_size, lattice_b);
}

/** mq_opt.pack_size walkers per cpu thread
 */
template<Devices DT>
template<Threading TT>
void MiniqmcDriverFunctions<DT>::crowd_thread_main(const int ip,
                                                   TaskBlockBarrier<TT>& barrier,
                                                   const int team_size,
                                                   MiniqmcOptions& mq_opt,
                                                   const PrimeNumberSet<uint32_t>& myPrimes,
                                                   ParticleSet ions,
                                                   const SPOSet* spo_main)
{
  const int member_id = ip % team_size;
  // create and initialize movers
  //app_summary() << "pack size:" << mq_opt.pack_size << '\n';
  app_summary() << "thread:" << ip << " starting up, with team_size: " << team_size << " member_id: " << member_id
                << ".\n";
  int my_accepts = 0;
  Crowd<DT> crowd(ip, myPrimes, ions, mq_opt.pack_size);

  crowd.init();
  // For VMC, tau is large and should result in an acceptance ratio of roughly
  // 50%
  // For DMC, tau is small and should result in an acceptance ratio of 99%

  const QMCT::RealType tau = 2.0;

  QMCT::RealType sqrttau = std::sqrt(tau);


  // create a spo view in each Mover
  crowd.buildViews(mq_opt.useRef, spo_main, team_size, member_id);

  crowd.buildWaveFunctions(mq_opt.useRef, mq_opt.enableJ3);

  // initial update
  std::for_each(crowd.elss_begin(), crowd.elss_end(), [](ParticleSet& els) { els.update(); });
  // for(auto& els_it = crowd.elss_begin(); els_it != crowd.elss_end(); els_it++)
  //   {
  //     els.epdate();
  //   }

  crowd.evaluateLog();

  //app_summary() << "initial update complete \n";

  const int nions = ions.getTotalNum();
  const int nels  = crowd.elss[0]->getTotalNum();
  const int nels3 = 3 * nels;

  // this is the number of quadrature points for the non-local PP
  const int nknots(crowd.nlpps[0]->size());

  for (int mc = 0; mc < mq_opt.nsteps; ++mc)
  {
    mq_opt.Timers[Timer_Diffusion]->start();

    for (int l = 0; l < mq_opt.nsubsteps; ++l) // drift-and-diffusion
    {
      crowd.fillRandoms();

      for (int iel = 0; iel < nels; ++iel)
      {
        // Operate on electron with index iel
        // probably should be in crowd
        std::for_each(crowd.elss_begin(), crowd.elss_end(), [iel](ParticleSet& els) { els.setActive(iel); });

        // Compute gradient at the current position
        mq_opt.Timers[Timer_evalGrad]->start();
        crowd.evaluateGrad(iel);
        mq_opt.Timers[Timer_evalGrad]->stop();

        crowd.constructTrialMoves(iel);

        // Compute gradient at the trial position
        mq_opt.Timers[Timer_ratioGrad]->start();
        crowd.evaluateRatioGrad(iel);
        mq_opt.Timers[Timer_ratioGrad]->stop();


        mq_opt.Timers[Timer_evalVGH]->start();
        crowd.evaluateHessian(iel);
        mq_opt.Timers[Timer_evalVGH]->stop();

        mq_opt.Timers[Timer_Update]->start();
        crowd.finishUpdate(iel);
        mq_opt.Timers[Timer_Update]->stop();


        // Accept/reject the trial move
        mq_opt.Timers[Timer_Update]->start();
        int these_accepts = crowd.acceptRestoreMoves(iel, mq_opt.accept);
        //app_summary() << "Moves accepted: " << these_accepts << "\n";
        my_accepts += these_accepts;
        mq_opt.Timers[Timer_Update]->stop();
      } //iel
    }   //substeps
    crowd.donePbyP();
    crowd.evaluateGL();

    mq_opt.Timers[Timer_ECP]->start();

    mq_opt.Timers[Timer_Value]->start();
    crowd.calcNLPP(nions, mq_opt.Rmax);
    mq_opt.Timers[Timer_Value]->stop();
    mq_opt.Timers[Timer_ECP]->stop();

    mq_opt.Timers[Timer_Diffusion]->stop();
  } // nsteps
  barrier.wait();
}

extern template class qmcplusplus::MiniqmcDriverFunctions<Devices::CPU>;

#ifdef QMC_USE_CUDA
extern template class qmcplusplus::MiniqmcDriverFunctions<Devices::CUDA>;
#endif

} // namespace qmcplusplus

#endif
