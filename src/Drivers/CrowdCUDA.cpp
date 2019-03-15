////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2018 QMCPACK developers.
//
// File developed by:
// Peter Doak, doakpw@ornl.gov, Oak Ridge National Lab
//
// File created by:
// Peter Doak, doakpw@ornl.gov, Oak Ridge National Lab
////////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <vector>
#include "Utilities/Configuration.h"
#include "Drivers/Crowd.hpp"
#include "QMCWaveFunctions/WaveFunction.h"
#include "QMCWaveFunctions/SPOSet_builder.h"
#include "Particle/ParticleSet_builder.hpp"
#include "Particle/ParticleSet.h"

namespace qmcplusplus
{

}

template<>
void Crowd<Devices::CUDA>::calcNLPP(int nions, QMCT::RealType Rmax)
{
  std::for_each(boost::make_zip_iterator(boost::make_tuple(wfs_begin(), spos.begin(), elss_begin(), nlpps_begin())),
                boost::make_zip_iterator(boost::make_tuple(wfs_end(), spos.end(), elss_end(), nlpps_end())),
                [nions, Rmax](const boost::tuple<WaveFunction&, SPOSet*, ParticleSet&, NonLocalPP<QMCT::RealType>&>& t) {
                  auto& wavefunction = t.get<0>();
                  auto& spo          = *(t.get<1>());
                  auto& els          = t.get<2>();
                  auto& ecp          = t.get<3>();
                  int nknots         = ecp.size();
                  ParticleSet::ParticlePos_t rOnSphere(nknots);
                  ecp.randomize(rOnSphere); // pick random sphere
                  const DistanceTableData* d_ie = els.DistTables[wavefunction.get_ei_TableID()];

                  for (int jel = 0; jel < els.getTotalNum(); ++jel)
                  {
                    const auto& dist  = d_ie->Distances[jel];
                    const auto& displ = d_ie->Displacements[jel];
                    for (int iat = 0; iat < nions; ++iat)
                      if (dist[iat] < Rmax)
                        for (int k = 0; k < nknots; k++)
                        {
                          QMCT::PosType deltar(dist[iat] * rOnSphere[k] - displ[iat]);
                          els.makeMoveOnSphere(jel, deltar);
                          spo.evaluate_v(els.R[jel]);
                          wavefunction.ratio(els, jel);
                          els.rejectMove(jel);
                        }
                  }
                });
}

template<>
void Crowd<Devices::CUDA>::evaluateHessian(int iel)
{
  spos.
  std::for_each(boost::make_zip_iterator(boost::make_tuple(spos.begin(), pos_list.begin())),
                boost::make_zip_iterator(boost::make_tuple(spos.end(), pos_list.end())),
                [&](const boost::tuple<SPOSet*, QMCT::PosType&>& t) { t.get<0>()->evaluate_vgh(t.get<1>()); });
}

template<Devices DT>
int Crowd<DT>::acceptRestoreMoves(int iel, QMCT::RealType accept)
{
  int accepted = 0;
  std::for_each(boost::make_zip_iterator(boost::make_tuple(urs_begin(), wfs_begin(), elss_begin())),
                boost::make_zip_iterator(boost::make_tuple(urs_end(), wfs_end(), elss_end())),
                [iel, accept, &accepted](const boost::tuple<aligned_vector<QMCT::RealType>&, WaveFunction&, ParticleSet&>& t) {
                  auto& els = t.get<2>();
                  if (t.get<0>()[iel] < accept)
                  {
		    ++accepted;
                    t.get<1>().acceptMove(els, iel);
                    els.acceptMove(iel);
                  }
                });
  return accepted;
}

  template<Devices DT>
  void Crowd<DT>::donePbyP()
  {
    std::for_each(elss_begin(), elss_end(), [](ParticleSet& els) { els.donePbyP(); });
  }
  
template<Devices DT>
void Crowd<DT>::fillRandoms()
{
  //We're going to generate many more random values than in the miniqmc_sync_move case
  std::for_each(boost::make_zip_iterator(boost::make_tuple(urs_begin(), rngs_begin())),
                boost::make_zip_iterator(boost::make_tuple(urs_end(), rngs_end())),
                [&](const boost::tuple<aligned_vector<QMCT::RealType>&, RandomGenerator<QMCT::RealType>&>& t) {
                  t.get<1>().generate_uniform(t.get<0>().data(), nels_);
                });
  //I think this means each walker/mover has the same stream of random values
  //it would if this was the threaded single implementation
  std::for_each(boost::make_zip_iterator(boost::make_tuple(deltas_begin(), rngs_begin())),
                boost::make_zip_iterator(boost::make_tuple(deltas_end(), rngs_end())),
                [&](const boost::tuple<std::vector<QMCT::PosType>&, RandomGenerator<QMCT::RealType>&>& t) {
                  t.get<1>().generate_normal(&(t.get<0>()[0][0]), nels_ * 3);
                });
}

template<Devices DT>
void Crowd<DT>::constructTrialMoves(int iel)
{
  std::for_each(boost::make_zip_iterator(boost::make_tuple(deltas_begin(), elss_begin(), valids.begin())),
                boost::make_zip_iterator(boost::make_tuple(deltas_end(), elss_end(), valids.end())),
                constructTrialMove(sqrttau, iel));
}

template class Crowd<Devices::CPU>;
#ifdef QMC_USE_KOKKOS
template class Crowd<Devices::KOKKOS>;
#endif
#ifdef QMC_USE_CUDA
template class Crowd<Devices::CUDA>;
#endif
} // namespace qmcplusplus
