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

#include "QMCWaveFunctions/SPOSet_builder.h"
#include "Devices.h"
#include "QMCWaveFunctions/WaveFunction.h"
#include "QMCWaveFunctions/SPOSet.h"
#include "QMCWaveFunctions/WaveFunctionBuilder.h"
#include "Input/pseudo.hpp"
#include "Utilities/RandomGenerator.h"
#include "Utilities/PrimeNumberSet.h"
#include "Drivers/CrowdBuffers.hpp"
#include "Memory/DeviceBuffers.hpp"

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

enum CrowdTimers
  {
    Timer_VGH,
    Timer_WF,
  };

    
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
    //   static constrexp Devices TDT = DT;
  //Functors for more complex things we want to do to each "mover"
  struct buildWaveFunctionsFunc
      : public std::unary_function<
      const boost::tuple<WaveFunction&, ParticleSet&, ParticleSet&, const RandomGenerator<QMCT::RealType>&, DeviceBuffers<DT>* >&, void>
  {
  private:
    bool useRef_;
    bool enableJ3_;
    ParticleSet& ions_;

  public:
      buildWaveFunctionsFunc(ParticleSet& ions, bool useRef = false, bool enableJ3 = false)
	  : useRef_(useRef), enableJ3_(enableJ3), ions_(ions)
          {}
      void operator()(const boost::tuple<WaveFunction&, ParticleSet&, const RandomGenerator<QMCT::RealType>&, DeviceBuffers<DT>*>& t) const
    {
	WaveFunctionBuilder<DT>::build(useRef_, t.get<0>(), ions_, t.get<1>(), t.get<2>(), t.get<3>(),  enableJ3_);
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

  void init();
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
  void evaluateLaplacian(int iel);
  void fillRandoms();
  void donePbyP();
  void constructTrialMoves(int iels);
  void updatePosFromCurrentEls(int iels);
  int acceptRestoreMoves(int iels, QMCT::RealType accept);

    void setupTimers() {
	  TimerNameLevelList_t<CrowdTimers> CrowdTimerNames =
    {{Timer_VGH, "Actual VGH eval", timer_level_fine}, {Timer_WF, "WF after VGH", timer_level_fine}};

  setup_timers(timers, CrowdTimerNames);
    }

    TimerList_t timers;
private:
  CrowdBuffers<DT> buffers_;
  std::vector<DeviceBuffers<DT>> device_buffers_;
  int pack_size_;
  static constexpr QMCT::RealType tau = 2.0;
  QMCT::RealType sqrttau = std::sqrt(tau);
  int nels_;
};

// This could all go in the .cpp but explicit instantiations of many functions would be necessary (I think)
// For now it gets dumped here

template<Devices DT>
Crowd<DT>::Crowd(const int ip, const PrimeNumberSet<uint32_t>& myPrimes, const ParticleSet& ions, const int pack_size)
    : ions_(ions),
      pack_size_(pack_size),
      pos_list(pack_size),
      grad_now(pack_size),
      grad_new(pack_size),
      ratios(pack_size),
      valids(pack_size)
{
  for (int i = 0; i < pack_size_; i++)
  {
    int prime_index = ip * pack_size_ + i;
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
  device_buffers_.resize(pack_size);
}

template<Devices DT>
template<Devices ODT>
Crowd<DT>::Crowd(const Crowd<ODT>& m)
{
  ions_     = m.ions_;
  rngs_     = m.rngs_; //Takes possessiong of ptrs?
}

template<Devices DT>
Crowd<DT>::~Crowd()
{}

template<Devices DT>
void Crowd<DT>::init()
{
}


  //This std::for_each idiom will make trying hpx threads very easy.
  
template<Devices DT>
void Crowd<DT>::updatePosFromCurrentEls(int iel)
{
  std::for_each(boost::make_zip_iterator(boost::make_tuple(elss_begin(), pos_list.begin())),
                boost::make_zip_iterator(boost::make_tuple(elss_end(), pos_list.end())),
                [iel](const boost::tuple<ParticleSet&, QMCT::PosType&>& t) { t.get<1>() = t.get<0>().R[iel]; });
}

template<Devices DT>
void Crowd<DT>::buildViews(bool useRef, const SPOSet* const spo_main, int team_size, int member_id)
{
  std::for_each(spos.begin(), spos.end(), [&](SPOSet*& s) {
    s = SPOSetBuilder<DT>::buildView(useRef, spo_main, team_size, member_id);
  });
}

template<Devices DT>
void Crowd<DT>::buildWaveFunctions(bool useRef, bool enableJ3)
{
    std::vector<DeviceBuffers<DT>*> ref_vec;
    for(int i = 0; i < device_buffers_.size(); ++i)
    {
	ref_vec.push_back(&device_buffers_[i]);
    }
    
    std::for_each(boost::make_zip_iterator(boost::make_tuple(wfs_begin(), elss_begin(), rngs_begin(), ref_vec.begin())),
		  boost::make_zip_iterator(boost::make_tuple(wfs_end(), elss_end(), rngs_end(), ref_vec.end())),
		  buildWaveFunctionsFunc(ions_, useRef , enableJ3));
}

template<Devices DT>
void Crowd<DT>::evaluateLog()
{
  std::for_each(boost::make_zip_iterator(boost::make_tuple(wfs_begin(), elss_begin())),
                boost::make_zip_iterator(boost::make_tuple(wfs_end(), elss_end())),
                [](const boost::tuple<WaveFunction&, ParticleSet&>& t) { t.get<0>().evaluateLog(t.get<1>()); });
}

template<Devices DT>
void Crowd<DT>::evaluateGrad(int iel)
{}

template<Devices DT>
void Crowd<DT>::evaluateRatioGrad(int iel)
{
  std::for_each(boost::make_zip_iterator(
                    boost::make_tuple(wfs_begin(), elss_begin(), ratios.begin(), grad_new.begin())),
                boost::make_zip_iterator(boost::make_tuple(wfs_end(), elss_end(), ratios.end(), grad_new.end())),
                [&](const boost::tuple<WaveFunction&, ParticleSet&, QMCT::ValueType&, QMCT::GradType&>& t) {
                  t.get<2>() = t.get<0>().ratioGrad(t.get<1>(), iel, t.get<3>());
                });
}

template<Devices DT>
void Crowd<DT>::evaluateGL()
{
  std::for_each(boost::make_zip_iterator(boost::make_tuple(wfs_begin(), elss_begin())),
                boost::make_zip_iterator(boost::make_tuple(wfs_end(), elss_end())),
                [&](const boost::tuple<WaveFunction&, ParticleSet&>& t) { t.get<0>().evaluateGL(t.get<1>()); });
}

template<Devices DT>
void Crowd<DT>::calcNLPP(int nions, QMCT::RealType Rmax)
{
  //std::cout << "Crowd calcNLPP Called \n"; 

  //So basically we need to unwrap this.

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

template<Devices DT>
void Crowd<DT>::evaluateHessian(int iel)
{
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



extern template class Crowd<Devices::CPU>;
#ifdef QMC_USE_KOKKOS
extern template class Crowd<Devices::KOKKOS>;
#endif
} // namespace qmcplusplus
#endif //QMCPLUSPLUS_MOVERS_HPP
