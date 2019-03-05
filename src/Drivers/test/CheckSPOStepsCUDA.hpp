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
#ifndef QMCPLUSPLUS_CHECK_SPO_STEPS_CUDA_HPP
#define QMCPLUSPLUS_CHECK_SPO_STEPS_CUDA_HPP

#include "Drivers/check_spo.h"
#include "Utilities/Configuration.h"

namespace qmcplusplus
{

template<>
template<typename T>
void CheckSPOSteps<Devices::CUDA>::thread_main(const int np,
                                    const int ip,
                                    const int team_size,
                                    const ParticleSet ions,
                                    const SPODevImp spo_main,
                                    const SPORef spo_ref_main,
                                    const int nsteps,
                                    const QMCT::RealType Rmax,
                                    T& ratio,
                                    T& nspheremoves,
                                    T& dNumVGHCalls,
                                    T& evalV_v_err,
                                    T& evalVGH_v_err,
                                    T& evalVGH_g_err,
                                    T& evalVGH_h_err)
{
  const int team_id   = ip / team_size;
  const int member_id = ip % team_size;

  // create generator within the thread
  RandomGenerator<QMCT::RealType> random_th(MakeSeed(team_id, np));

  ParticleSet els;
  build_els(els, ions, random_th);
  els.update();

  const int nions = ions.getTotalNum();
  const int nels  = els.getTotalNum();
  const int nels3 = 3 * nels;

  // create pseudopp
  NonLocalPP<OHMMS_PRECISION> ecp(random_th);
  xo// create spo per thread
  SPODevImp spo(spo_main, team_size,member_id);
  //SPODevImp& spo = *dynamic_cast<SPODevImp*>(SPOSetBuilder<DT>::buildView(false, spo_main, team_size, member_id));
  SPORef spo_ref(spo_ref_main, team_size, member_id);

  // use teams
  // if(team_size>1 && team_size>=nTiles ) spo.set_range(team_size,ip%team_size);

  // this is the cutoff from the non-local PP
  const int nknots(ecp.size());

  ParticleSet::ParticlePos_t delta(nels);
  ParticleSet::ParticlePos_t rOnSphere(nknots);

  QMCT::RealType sqrttau = 2.0;
  QMCT::RealType accept  = 0.5;

  vector<QMCT::RealType> ur(nels);
  random_th.generate_uniform(ur.data(), nels);
  const double zval = 1.0 * static_cast<double>(nels) / static_cast<double>(nions);

  int my_accepted = 0, my_vals = 0;

  EinsplineSPOParams<T> esp = spo.getParams();
  for (int mc = 0; mc < nsteps; ++mc)
  {
    random_th.generate_normal(&delta[0][0], nels3);
    random_th.generate_uniform(ur.data(), nels);
    for (int ib = 0; ib < esp.nBlocks; ib++)

      // VMC
      for (int iel = 0; iel < nels; ++iel)
      {
        QMCT::PosType pos = els.R[iel] + sqrttau * delta[iel];

        spo.evaluate_vgh(pos);
        spo_ref.evaluate_vgh(pos);
        // accumulate error
        for (int ib = 0; ib < esp.nBlocks; ib++)
          for (int n = 0; n < esp.nSplinesPerBlock; n++)
          {
            // value
            evalVGH_v_err += std::fabs(spo.getPsi(ib, n) - spo_ref.psi[ib][n]);
            // grad
            evalVGH_g_err += std::fabs(spo.getGrad(ib, n, 0) - spo_ref.grad[ib].data(0)[n]);
            evalVGH_g_err += std::fabs(spo.getGrad(ib, n, 1) - spo_ref.grad[ib].data(1)[n]);
            evalVGH_g_err += std::fabs(spo.getGrad(ib, n, 2) - spo_ref.grad[ib].data(2)[n]);
            // hess
            evalVGH_h_err += std::fabs(spo.getHess(ib, n, 0) - spo_ref.hess[ib].data(0)[n]);
            evalVGH_h_err += std::fabs(spo.getHess(ib, n, 1) - spo_ref.hess[ib].data(1)[n]);
            evalVGH_h_err += std::fabs(spo.getHess(ib, n, 2) - spo_ref.hess[ib].data(2)[n]);
            evalVGH_h_err += std::fabs(spo.getHess(ib, n, 3) - spo_ref.hess[ib].data(3)[n]);
            evalVGH_h_err += std::fabs(spo.getHess(ib, n, 4) - spo_ref.hess[ib].data(4)[n]);
            evalVGH_h_err += std::fabs(spo.getHess(ib, n, 5) - spo_ref.hess[ib].data(5)[n]);
          }
        if (ur[iel] < accept)
        {
          els.R[iel] = pos;
          my_accepted++;
        }
      }

    random_th.generate_uniform(ur.data(), nels);
    ecp.randomize(rOnSphere); // pick random sphere
    for (int iat = 0, kat = 0; iat < nions; ++iat)
    {
      const int nnF    = static_cast<int>(ur[kat++] * zval);
      QMCT::RealType r = Rmax * ur[kat++];
      auto centerP     = ions.R[iat];
      my_vals += (nnF * nknots);

      for (int nn = 0; nn < nnF; ++nn)
      {
        for (int k = 0; k < nknots; k++)
        {
          QMCT::PosType pos = centerP + r * rOnSphere[k];
          spo.evaluate_v(pos);
          spo_ref.evaluate_v(pos);
          // accumulate error
          for (int ib = 0; ib < esp.nBlocks; ib++)
            for (int n = 0; n < esp.nSplinesPerBlock; n++)
              evalV_v_err += std::fabs(spo.getPsi(ib, n) - spo_ref.psi[ib][n]);
        }
      } // els
    }   // ions

  } // steps.

  ratio += QMCT::RealType(my_accepted) / QMCT::RealType(nels * nsteps);
  nspheremoves += QMCT::RealType(my_vals) / QMCT::RealType(nsteps);
  dNumVGHCalls += nels;
}

extern template class CheckSPOSteps<Devices::CUDA>;
} // namespace qmcplusplus

#endif
