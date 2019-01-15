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

#ifndef QMCPLUSPLUS_WALKERS_HPP
#define QMCPLUSPLUS_WALKERS_HPP

#include<memory>
#include<functional>
#include<boost/tuple/tuple.hpp>
#include<boost/iterator/zip_iterator.hpp>

#include "Devices.h"
#include "QMCWaveFunctions/WaveFunction.h"
#include "QMCWaveFunctions/SPOSet.h"
#include "Input/pseudo.hpp"
#include "Utilities/RandomGenerator.h"
#include "Utilities/PrimeNumberSet.h"
namespace qmcplusplus
{

/** Mover collection does not use mover as atom
 *  This is because extract list functions of mover demonstrate this isn't useful.
 */

template<Devices DT>
class Movers
{
  using QMCT = QMCTraits;
  struct buildWaveFunctionsFunc :  
    public std::unary_function<const boost::tuple<WaveFunction&, ParticleSet&, ParticleSet&, const RandomGenerator<QMCT::RealType>&>&, void>
  {
  private:
    bool useRef_;
    bool enableJ3_;
    ParticleSet& ions_;
  public:
    buildWaveFunctionsFunc(ParticleSet& ions,
		  bool useRef= false,
		  bool enableJ3= false) : ions_(ions), useRef_(useRef), enableJ3_(enableJ3) {}
    void operator() (const boost::tuple<WaveFunction&, ParticleSet&, ParticleSet&, const RandomGenerator<QMCT::RealType>&>& t) const
    {
      WaveFunctionBuilder<DT>::build(useRef_,
				     t.get<0>(),
				     ions_,
				     t.get<1>(),
				     t.get<2>(),
				     enableJ3_);
    }
  };
public:
  ParticleSet ions_;
  /// random number generator
  std::vector<std::unique_ptr<RandomGenerator<QMCT::RealType>>> rngs_;
  /// electrons
  std::vector<std::unique_ptr<ParticleSet>> elss;
  /// single particle orbitals
  std::vector<SPOSet*> spos;
  /// wavefunction container
  std::vector<std::unique_ptr<WaveFunction>> wavefunctions;
  /// non-local pseudo-potentials
  std::vector<std::unique_ptr<NonLocalPP<QMCT::RealType>>> nlpps;

  /// constructor
  Movers(const PrimeNumberSet<uint32_t>& myPrimes,
	  const ParticleSet& ions,
	 const int pack_size);

  /// destructor
  ~Movers();
  void buildViews(bool useRef,
	          const SPOSet* spo_main,
		  int team_size,
		  int member_id);

  void buildWaveFunctions(bool useRef,
			  bool enableJ3);

  
};

}
#endif //QMCPLUSPLUS_MOVERS_HPP
