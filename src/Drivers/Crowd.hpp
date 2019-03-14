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

#include <memory>
#include <functional>
#include <type_traits>
#include <stdexcept>
#include <boost/tuple/tuple.hpp>
#include <boost/iterator/zip_iterator.hpp>

#include "Devices.h"
#include "QMCWaveFunctions/WaveFunction.h"
#include "QMCWaveFunctions/SPOSet.h"
#include "Input/pseudo.hpp"
#include "Utilities/RandomGenerator.h"
#include "Utilities/PrimeNumberSet.h"
namespace qmcplusplus
{

/** exercise for the reader
 *  works for std::algorithms
 *  But not with standard iterator loop syntax
 */

template<class BaseIterator>
class DereferenceIterator : public BaseIterator
{
public:
  using value_type = typename BaseIterator::value_type::element_type;
  using pointer    = value_type*;
  using reference  = value_type&;

  DereferenceIterator(const BaseIterator& other) : BaseIterator(other) {}

  reference operator*() const { return *(this->BaseIterator::operator*()); }
  pointer operator->() const { return this->BaseIterator::operator*().get(); }
  // reference operator++() const {return *(this->BaseIterator::operator++()); }
  reference operator[](size_t n) const { return *(this->BaseIterator::operator[](n)); }
};

template<typename Iterator>
DereferenceIterator<Iterator> dereference_iterator(Iterator t)
{
  return DereferenceIterator<Iterator>(t);
}

/** Mover collection does not use mover as atom
 *  This is because extract list functions of mover demonstrate this isn't useful.
 *  It always does packsize batching, even on invalid moves,
 *  The assumption is with batching there is no advantage to dropping evals of invalid
 *  if you are faking batching serially this obviously has cost.
 *  Except for testing just thread with pack_size 1 movers
 */

template<Devices DT>
class Crowd
{
  using QMCT = QMCTraits;

  //Functors for more complex things we want to do to each "mover"
  struct buildWaveFunctionsFunc
      : public std::unary_function<
            const boost::tuple<WaveFunction&, ParticleSet&, ParticleSet&, const RandomGenerator<QMCT::RealType>&>&,
            void>
  {
  private:
    bool useRef_;
    bool enableJ3_;
    ParticleSet& ions_;

  public:
    buildWaveFunctionsFunc(ParticleSet& ions, bool useRef = false, bool enableJ3 = false)
      : useRef_(useRef), enableJ3_(enableJ3), ions_(ions)
    {}
    void operator()(const boost::tuple<WaveFunction&, ParticleSet&, const RandomGenerator<QMCT::RealType>&>& t) const
    {
      WaveFunctionBuilder<DT>::build(useRef_, t.get<0>(), ions_, t.get<1>(), t.get<2>(), enableJ3_);
    }
  };

  struct constructTrialMove
    : public std::unary_function<const boost::tuple<std::vector<QMCT::PosType>&,
						    ParticleSet&,
						    int&>&, void>
  {
    int iel_;
    QMCT::RealType sqrttau_;
    constructTrialMove(QMCT::RealType sqrttau, int iel)
      : iel_(iel), sqrttau_(sqrttau)
    {}
    void operator()(const boost::tuple<std::vector<QMCT::PosType>&,
		    ParticleSet&,
		    int&>& t) const
    {
      QMCT::PosType dr = sqrttau_ * t.get<0>()[iel_];
      int isValid = t.get<1>().makeMoveAndCheck(iel_, dr);
      if(isValid == 0)
	throw std::logic_error("moves must be valid");
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

  std::vector<std::unique_ptr<std::vector<QMCT::PosType>>> deltas;
  std::vector<QMCT::PosType> pos_list;
  std::vector<QMCT::GradType> grad_now;
  std::vector<QMCT::GradType> grad_new;
  std::vector<QMCT::ValueType> ratios;
  std::vector<int> valids;
  std::vector<std::unique_ptr<aligned_vector<QMCT::RealType>>> urs;

  //These let outsiders touch the objects in the pointer vectors
  //There is a smell to this happening so much
  //On the other hand object lifetime and ownership is being respected
  //preventing double frees, invalid objects and leaking
  DereferenceIterator<std::vector<std::unique_ptr<ParticleSet>>::iterator> elss_begin()
  {
    return dereference_iterator(elss.begin());
  }
  DereferenceIterator<std::vector<std::unique_ptr<ParticleSet>>::iterator> elss_end()
  {
    return dereference_iterator(elss.end());
  }
  DereferenceIterator<std::vector<std::unique_ptr<WaveFunction>>::iterator> wfs_begin()
  {
    return dereference_iterator(wavefunctions.begin());
  }
  DereferenceIterator<std::vector<std::unique_ptr<WaveFunction>>::iterator> wfs_end()
  {
    return dereference_iterator(wavefunctions.end());
  }
  DereferenceIterator<std::vector<std::unique_ptr<RandomGenerator<QMCT::RealType>>>::iterator> rngs_begin()
  {
    return dereference_iterator(rngs_.begin());
  }
  DereferenceIterator<std::vector<std::unique_ptr<RandomGenerator<QMCT::RealType>>>::iterator> rngs_end()
  {
    return dereference_iterator(rngs_.end());
  }
  DereferenceIterator<std::vector<std::unique_ptr<aligned_vector<QMCT::RealType>>>::iterator> urs_begin()
  {
    return dereference_iterator(urs.begin());
  }
  DereferenceIterator<std::vector<std::unique_ptr<aligned_vector<QMCT::RealType>>>::iterator> urs_end()
  {
    return dereference_iterator(urs.end());
  }
  DereferenceIterator<std::vector<std::unique_ptr<std::vector<QMCT::PosType>>>::iterator> deltas_begin()
  {
    return dereference_iterator(deltas.begin());
  }
  DereferenceIterator<std::vector<std::unique_ptr<std::vector<QMCT::PosType>>>::iterator> deltas_end()
  {
    return dereference_iterator(deltas.end());
  }
  DereferenceIterator<std::vector<std::unique_ptr<NonLocalPP<QMCT::RealType>>>::iterator> nlpps_begin()
  {
    return dereference_iterator(nlpps.begin());
  }
  DereferenceIterator<std::vector<std::unique_ptr<NonLocalPP<QMCT::RealType>>>::iterator> nlpps_end()
  {
    return dereference_iterator(nlpps.end());
  }

  // DereferenceIterator<std::vector<std::shared_ptr<SPOSet>>::iterator> spos_begin()
  // {
  //   return dereference_iterator(spos.begin());
  // }
  // DereferenceIterator<std::vector<std::shared_ptr<SPOSet>>::iterator> spos_end()
  // {
  //   return dereference_iterator(spos.end());
  // }

  
  ParticleSet& elss_back() { return *(elss.back()); }

  RandomGenerator<QMCT::RealType>& rngs_back() { return *rngs_.back(); }

  /// constructor
  Crowd(const int ip, const PrimeNumberSet<uint32_t>& myPrimes, const ParticleSet& ions, const int pack_size);

  /** conversion from one dev
    *   to another
    */
  template<Devices ODT>
  Crowd(const Crowd<ODT>& m);
  
  /// destructor
  ~Crowd();

  /** takes over the 'movers' of passed in mover
   *  that object becomes invalid
   */
  void merge(Crowd& m_in) {} 
  void buildViews(bool useRef, const SPOSet* const spo_main, int team_size, int member_id);

  void buildWaveFunctions(bool useRef, bool enableJ3);

  void evaluateLog();
  void evaluateGrad(int iel);
  void evaluateGL();
  void calcNLPP(int nions, QMCT::RealType Rmax);
  void evaluateRatioGrad(int iel);
  void evaluateHessian(int iel);
  void fillRandoms();
  void donePbyP();
  void constructTrialMoves(int iels);
  void updatePosFromCurrentEls(int iels);
  int acceptRestoreMoves(int iels, QMCT::RealType accept);
private:
  int pack_size_;
  static constexpr QMCT::RealType tau = 2.0;
  QMCT::RealType sqrttau = std::sqrt(tau);
  int nels_;
};

extern template class Crowd<Devices::CPU>;
#ifdef QMC_USE_KOKKOS
extern template class Crowd<Devices::KOKKOS>;
#endif
#ifdef QMC_USE_CUDA
extern template class Crowd<Devices::CUDA>;
#endif 
} // namespace qmcplusplus
#endif //QMCPLUSPLUS_MOVERS_HPP
