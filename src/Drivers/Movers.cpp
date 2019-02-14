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
#include "Drivers/Movers.hpp"
#include "QMCWaveFunctions/WaveFunction.h"
#include "QMCWaveFunctions/SPOSet_builder.h"
#include "Particle/ParticleSet_builder.hpp"
namespace qmcplusplus
{
template<Devices DT>
Movers<DT>::Movers(const int ip, const PrimeNumberSet<uint32_t>& myPrimes, const ParticleSet& ions, const int pack_size)
  : ions_(ions), pack_size_(pack_size), pos_list(pack_size), grad_now(pack_size), grad_new(pack_size), ratios(pack_size), valids(pack_size)
{
  for (int i = 0; i < pack_size_; i++)
  {
    int prime_index =
      ip * pack_size_ + i;
    rngs_.push_back(std::make_unique<RandomGenerator<QMCT::RealType>>(myPrimes[prime_index]));
    nlpps.push_back(std::make_unique<NonLocalPP<QMCT::RealType>>(*rngs_.back()));
    elss.push_back(std::make_unique<ParticleSet>());
    wavefunctions.push_back(std::make_unique<WaveFunction>());
    //seems a bit inconsistent to go back to this pattern
    spos.push_back(nullptr);
    nels_ = build_els(elss_back(), ions_, rngs_back());
    urs.push_back(std::make_unique<aligned_vector<QMCT::RealType>>(nels_));
    deltas.push_back(std::make_unique<std::vector<QMCT::PosType>>(3 * nels_));
  }
}

template<Devices DT>
Movers<DT>::~Movers()
{

}

template<Devices DT>
void Movers<DT>::updatePosFromCurrentEls(int iel)
{
  std::for_each(boost::make_zip_iterator(boost::make_tuple(elss_begin(), pos_list.begin())),
                boost::make_zip_iterator(boost::make_tuple(elss_end(), pos_list.end())),
                [iel](const boost::tuple<ParticleSet&, QMCT::PosType&>& t)
		{
		  t.get<1>() = t.get<0>().R[iel];
		});
}

template<Devices DT>
void Movers<DT>::buildViews(bool useRef, std::shared_ptr<SPOSet> spo_main, int team_size, int member_id)
{
  std::for_each(spos.begin(), spos.end(), [&](std::shared_ptr<SPOSet> s) {
    s = SPOSetBuilder<DT>::buildView(useRef, spo_main, team_size, member_id);
  });
}

template<Devices DT>
void Movers<DT>::buildWaveFunctions(bool useRef, bool enableJ3)
{
  std::for_each(boost::make_zip_iterator(boost::make_tuple(wfs_begin(), elss_begin(), rngs_begin())),
                boost::make_zip_iterator(boost::make_tuple(wfs_end(), elss_end(), rngs_end())),
                buildWaveFunctionsFunc(ions_, useRef, enableJ3));
}

template<Devices DT>
void Movers<DT>::evaluateLog()
{
  std::for_each(boost::make_zip_iterator(boost::make_tuple(wfs_begin(), elss_begin())),
                boost::make_zip_iterator(boost::make_tuple(wfs_end(), elss_end())),
                [](const boost::tuple<WaveFunction&, ParticleSet&>& t)
  {
    t.get<0>().evaluateLog(t.get<1>());
  });
}

template<Devices DT>
void Movers<DT>::evaluateGrad(int iel)
{
}

template<Devices DT>
void Movers<DT>::evaluateRatioGrad(int iel)
{
    std::for_each(boost::make_zip_iterator(boost::make_tuple(wfs_begin(), elss_begin(),
							   ratios.begin(),grad_new.begin())),
                  boost::make_zip_iterator(boost::make_tuple(wfs_end(), elss_end(),
							   ratios.end(),grad_new.end())),
                [&](const boost::tuple<WaveFunction&, ParticleSet&,
		   QMCT::ValueType&, QMCT::GradType&>& t)
		  { t.get<2>() = t.get<0>().ratioGrad(t.get<1>(), iel, t.get<3>()); });

}

template<Devices DT>
void Movers<DT>::evaluateHessian(int iel)
{
  std::for_each(boost::make_zip_iterator(boost::make_tuple(spos_begin(), pos_list.begin())),
		boost::make_zip_iterator(boost::make_tuple(spos_end(), pos_list.end())),
                [&](const boost::tuple<SPOSet&, QMCT::PosType&>& t)
		{ t.get<0>().evaluate_vgh(t.get<1>()); });

}

template<Devices DT>
void Movers<DT>::fillRandoms()
{
    //We're going to generate many more random values than in the miniqmc_sync_move case
  std::for_each(boost::make_zip_iterator(boost::make_tuple(urs_begin(), rngs_begin())),
		boost::make_zip_iterator(boost::make_tuple(urs_end(), rngs_end())),
		[&](const boost::tuple<aligned_vector<QMCT::RealType>&, RandomGenerator<QMCT::RealType>&>& t)
		{
		  t.get<1>().generate_uniform(t.get<0>().data(), nels_);
		});
  //I think this means each walker/mover has the same stream of random values
  //it would if this was the threaded single implementation
  std::for_each(boost::make_zip_iterator(boost::make_tuple(deltas_begin(), rngs_begin())),
    boost::make_zip_iterator(boost::make_tuple(deltas_end(), rngs_end())),
		[&](const boost::tuple<std::vector<QMCT::PosType>&, RandomGenerator<QMCT::RealType>&>& t)
		{
		  t.get<1>().generate_normal(&(t.get<0>()[0][0]), nels_ * 3);
		}
		);
}
  
template<Devices DT>
void Movers<DT>::constructTrialMoves(int iel)
{
  std::for_each(boost::make_zip_iterator(boost::make_tuple(deltas_begin(),elss_begin(),valids.begin())),
		boost::make_zip_iterator(boost::make_tuple(deltas_end(),elss_end(),valids.end())),
		constructTrialMove(sqrttau, iel));
  
}
  
template class Movers<Devices::CPU>;
} // namespace qmcplusplus
