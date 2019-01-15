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
namespace qmcplusplus
{

template<Devices DT>
Movers<DT>::Movers(const PrimeNumberSet<uint32_t>& myPrimes,
	  const ParticleSet& ions,
          const int pack_size) : ions_(ions)
  {
    // A thread safe std::algorithm approach would be better
    // what would it take to keep things determinstic with rngs_
    for(int i = 0; i < pack_size; i++)
    {
      rngs_.push_back(std::make_unique<RandomGenerator<QMCT::RealType>>(myPrimes[i]));
      nlpps.push_back(std::make_unique<NonLocalPP<QMCT::RealType>>(rngs_.back()));
      elss.push_back(std::make_unique<ParticleSet>());
      wavefunctions.push_back(std::make_unique<WaveFunction>());
      //seems a bit inconsistent to go back to this pattern
      spos.push_back(nullptr);
      build_els(elss.back(), ions_, rngs_.back());
    }
  }

  template<Devices DT>
  Movers<DT>::~Movers()
  {
    std::for_each(spos.begin(), spos.end(), [](SPOSet* s)
		  {
		    if(s!=nullptr)
		    {
		      delete s;
		    }
		  });
  }

  template<Devices DT>
  void Movers<DT>::buildViews(bool useRef,
			      const SPOSet* spo_main,
			      int team_size,
			      int member_id)
  {
    std::for_each(spos.begin(), spos.end(), [&](SPOSet* s)
		  {
		    s = SPOSetBuilder<DT>::buildView(useRef, spo_main, team_size, member_id);
		  });
  }
  template<Devices DT>
  void Movers<DT>::buildWaveFunctions(bool useRef,
				      bool enableJ3)
  {
    std::vector<std::unique_ptr<WaveFunction>>::const_iterator beg_wf = wavefunctions.begin();
    std::vector<std::unique_ptr<ParticleSet>>::const_iterator beg_els = elss.begin();
    std::vector<std::unique_ptr<RandomGenerator<QMCT::RealType>>>::const_iterator beg_rng = rngs_.begin();
    std::vector<std::unique_ptr<WaveFunction>>::const_iterator end_wf = wavefunctions.end();
    std::vector<std::unique_ptr<ParticleSet>>::const_iterator end_els = elss.end();
    std::vector<std::unique_ptr<RandomGenerator<QMCT::RealType>>>::const_iterator end_rng = rngs_.end();

    buildWaveFunctionsFunc build_func(ions_, useRef, enableJ3);
    std::for_each(boost::make_zip_iterator(boost::make_tuple(beg_wf, beg_els, beg_rng)),
		  boost::make_zip_iterator(boost::make_tuple(end_wf, end_els, end_rng)),
		  build_func());
  }
}
