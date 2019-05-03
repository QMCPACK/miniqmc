//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License. See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by: Jeongnim Kim, jeongnim.kim@intel.com, Intel Corp.
//                    Amrita Mathuriya, amrita.mathuriya@intel.com, Intel Corp.
//                    Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//
// File created by: Jeongnim Kim, jeongnim.kim@intel.com, Intel Corp.
//////////////////////////////////////////////////////////////////////////////////////
// -*- C++ -*-
#ifndef QMCPLUSPLUS_TWOBODYJASTROW_H
#define QMCPLUSPLUS_TWOBODYJASTROW_H
#include "Utilities/Configuration.h"
#include "QMCWaveFunctions/WaveFunctionKokkos.h"
#include "QMCWaveFunctions/WaveFunctionComponent.h"
#include "Particle/DistanceTableData.h"
#include "QMCWaveFunctions/Jastrow/TwoBodyJastrowKokkos.h"
#include <Utilities/SIMD/allocator.hpp>
#include <Utilities/SIMD/algorithm.hpp>
#include <numeric>

/*!
 * @file TwoBodyJastrow.h
 */

namespace qmcplusplus
{
/** @ingroup WaveFunctionComponent
 *  @brief Specialization for two-body Jastrow function using multiple functors
 *
 * Each pair-type can have distinct function \f$u(r_{ij})\f$.
 * For electrons, distinct pair correlation functions are used
 * for spins up-up/down-down and up-down/down-up.
 *
 * Based on TwoBodyJastrow.h with these considerations
 * - DistanceTableData using SoA containers
 * - support mixed precision: FT::real_type != OHMMS_PRECISION
 * - loops over the groups: elminated PairID
 * - support simd function
 * - double the loop counts
 * - Memory use is O(N).
 */

template<typename atbjdType, typename apsdType>
void doTwoBodyJastrowMultiEvaluateGL(atbjdType atbjd, apsdType apsd, bool fromscratch) {
  const int numWalkers = atbjd.extent(0);
  using BarePolicy = Kokkos::TeamPolicy<>;
  BarePolicy pol(numWalkers, 1, 32);
  Kokkos::parallel_for("tbj-evalGL-waker-loop", pol,
		       KOKKOS_LAMBDA(BarePolicy::member_type member) {
			 int walkerNum = member.league_rank(); 
			 atbjd(walkerNum).evaluateGL(member, apsd(walkerNum), fromscratch);
		       });
}

template<typename atbjdType, typename apsdType>
  void doTwoBodyJastrowMultiAcceptRestoreMove(atbjdType atbjd, apsdType apsd, 
					      Kokkos::View<int*>& isAcceptedMap,
					      int numAccepted, int iat) {
  const int numWalkers = numAccepted;
  using BarePolicy = Kokkos::TeamPolicy<>;
  BarePolicy pol(numWalkers, 1, 32);
  Kokkos::parallel_for("tbj-acceptRestoreMove-waker-loop", pol,
		       KOKKOS_LAMBDA(BarePolicy::member_type member) {
			 int walkerIdx = member.league_rank(); 
			 const int walkerNum = isAcceptedMap(walkerIdx);
			 atbjd(walkerNum).acceptMove(member, apsd(walkerNum), iat);
		       });
}


template<typename atbjdType, typename apsdType>
  void doTwoBodyJastrowMultiAcceptRestoreMove(atbjdType atbjd, apsdType apsd, int iat) {
  const int numWalkers = atbjd.extent(0);
  using BarePolicy = Kokkos::TeamPolicy<>;
  BarePolicy pol(numWalkers, 16, 32);
  Kokkos::parallel_for("tbj-acceptRestoreMove-waker-loop", pol,
		       KOKKOS_LAMBDA(BarePolicy::member_type member) {
			 int walkerNum = member.league_rank(); 
			 atbjd(walkerNum).acceptMove(member, apsd(walkerNum), iat);
		       });
}


template<typename atbjdType, typename apsdType, typename valT>
void doTwoBodyJastrowMultiRatioGrad(atbjdType& atbjd, apsdType& apsd, Kokkos::View<int*>& isValidMap,
				int numValid, int iel, Kokkos::View<valT**> gradNowView,
				Kokkos::View<valT*> ratiosView) {
  const int numWalkers = numValid;
  using BarePolicy = Kokkos::TeamPolicy<>;
  BarePolicy pol(numWalkers, 1, 32);
  Kokkos::parallel_for("obj-evalRatioGrad-walker-loop", pol,
		       KOKKOS_LAMBDA(BarePolicy::member_type member) {
			 int walkerIdx = member.league_rank();
			 int walkerNum = isValidMap(walkerIdx);
			 auto gv = Kokkos::subview(gradNowView,walkerIdx,Kokkos::ALL());
			 ratiosView(walkerIdx) = atbjd(walkerNum).ratioGrad(member, apsd(walkerNum), iel, gv);
		       });
}

template<typename atbjdType, typename apsdType, typename valT>
void doTwoBodyJastrowMultiRatioGrad(atbjdType atbjd, apsdType apsd, int iat, 
				    Kokkos::View<valT**> gradNowView,
				    Kokkos::View<valT*> ratiosView) {
  const int numWalkers = atbjd.extent(0);
  using BarePolicy = Kokkos::TeamPolicy<>;
  BarePolicy pol(numWalkers, 1, 32);
  Kokkos::parallel_for("tbj-evalRatioGrad-walker-loop", pol,
		       KOKKOS_LAMBDA(BarePolicy::member_type member) {
			 int walkerNum = member.league_rank();
			 auto gv = Kokkos::subview(gradNowView,walkerNum,Kokkos::ALL());
			 ratiosView(walkerNum) = atbjd(walkerNum).ratioGrad(member, apsd(walkerNum), iat, gv);
		       });
}
			 
template<typename atbjdType, typename valT>
void doTwoBodyJastrowMultiEvalGrad(atbjdType atbjd, int iat, Kokkos::View<valT**> gradNowView) {
  int numWalkers = atbjd.extent(0);
  using BarePolicy = Kokkos::TeamPolicy<>;
  BarePolicy pol(numWalkers, 1, 32);
  Kokkos::parallel_for("tbj-evalGrad-walker-loop", pol,
		       KOKKOS_LAMBDA(BarePolicy::member_type member) {
			 int walkerNum = member.league_rank();
			 for (int idim = 0; idim < gradNowView.extent(1); idim++) {
			   gradNowView(walkerNum,idim) = atbjd(walkerNum).dUat(iat,idim);
			 }
		       });
 }


template<typename eiListType, typename apskType, typename atbjdType, typename tempRType,
  typename walkerIdType, typename devRatioType>
void doTwoBodyJastrowMultiEvalRatio(int pairNum, eiListType& eiList, apskType& apsk,
				    atbjdType& allTwoBodyJastrowData,
				    tempRType& likeTempR, walkerIdType& activeWalkerIdx,
				    devRatioType& devRatios) {
  int numWalkers = activeWalkerIdx.extent(0);
  int numKnots = likeTempR.extent(1);

  using BarePolicy = Kokkos::TeamPolicy<>;
  BarePolicy pol(numWalkers, Kokkos::AUTO, 32);
  
  Kokkos::parallel_for("tbj-multi-ratio", pol,
		       KOKKOS_LAMBDA(BarePolicy::member_type member) {
			 int walkerIndex = member.league_rank();
			 int walkerNum = activeWalkerIdx(walkerIndex);
			 auto& psk = apsk(walkerNum);
			 //auto& jd = allTwoBodyJastrowData(walkerIndex);
			 allTwoBodyJastrowData(walkerIndex).updateMode(0) = 0;

			 Kokkos::parallel_for(Kokkos::TeamThreadRange(member, numKnots),
					      [=](const int& knotNum) {
						auto singleDists = Kokkos::subview(likeTempR, walkerNum, knotNum, Kokkos::ALL);
						int iel = eiList(walkerNum, pairNum, 0);
						auto val = allTwoBodyJastrowData(walkerIndex).computeU(member, psk, iel, singleDists);
						devRatios(walkerIndex, numKnots) = std::exp(allTwoBodyJastrowData(walkerIndex).Uat(iel) - val);
					      });
		       });

}

 
template<typename atbjdType, typename apsdType, typename valT>
void doTwoBodyJastrowMultiEvaluateLog(atbjdType atbjd, apsdType apsd, Kokkos::View<valT*> values) {
  Kokkos::Profiling::pushRegion("2BJ-multiEvalLog");
  const int numWalkers = atbjd.extent(0);
  using BarePolicy = Kokkos::TeamPolicy<>;
  const int numElectrons = atbjd(0).Nelec(0);


  BarePolicy pol(numWalkers*numElectrons, 8, 32);
  Kokkos::parallel_for("tbj-evalLog-waker-loop", pol,
		       KOKKOS_LAMBDA(BarePolicy::member_type member) {
			 int walkerNum = member.league_rank()/numElectrons;
			 values(walkerNum) = atbjd(walkerNum).evaluateLog(member, apsd(walkerNum));
		       });
  Kokkos::Profiling::popRegion();
}








template<class FT>
struct TwoBodyJastrow : public WaveFunctionComponent
{
  using jasDataType = TwoBodyJastrowKokkos<RealType, ValueType, OHMMS_DIM>;
  jasDataType jasData;
  bool splCoefsNotAllocated;
  
#ifdef QMC_PARALLEL_JASTROW
  typedef Kokkos::TeamPolicy<> policy_t;
#else
  typedef Kokkos::TeamPolicy<Kokkos::Serial> policy_t;
#endif
  /// alias FuncType
  using FuncType = FT;
  /// type of each component U, dU, d2U;
  using valT = typename FT::real_type;
  /// element position type
  using posT = TinyVector<valT, OHMMS_DIM>;
  /// use the same container
  using RowContainer = DistanceTableData::RowContainer;

  /// number of particles
  size_t N;
  /// number of particles + padded
  size_t N_padded;
  /// number of groups of the target particleset
  size_t NumGroups;
  /// Used to compute correction
  bool FirstTime;
  /// diff value
  RealType DiffVal;
  /// Correction
  RealType KEcorr;
  ///\f$Uat[i] = sum_(j) u_{i,j}\f$
  Vector<valT> Uat;
  ///\f$dUat[i] = sum_(j) du_{i,j}\f$
  using gContainer_type = VectorSoAContainer<valT, OHMMS_DIM>;
  gContainer_type dUat;
  ///\f$d2Uat[i] = sum_(j) d2u_{i,j}\f$
  Vector<valT> d2Uat;
  valT cur_Uat;
  Kokkos::View<valT*> cur_u, cur_du, cur_d2u;
  Kokkos::View<valT*> old_u, old_du, old_d2u;
  Kokkos::View<valT*> DistCompressed;
  Kokkos::View<int*> DistIndice;
  /// Container for \f$F[ig*NumGroups+jg]\f$
  typedef Kokkos::Device<
            Kokkos::DefaultHostExecutionSpace,
            typename Kokkos::DefaultExecutionSpace::memory_space>
    F_device_type;

  Kokkos::View<FT*,F_device_type> F;
  /// Uniquue J2 set for cleanup
//  std::map<std::string, FT*> J2Unique;

  //These are needed because the kokkos class function operator() for
  //parallel exeuction doesn't really take arguments...  just has
  //access to class variables.  
  int iat, igt, jg_hack;
  const RealType* dist;
  RealType* u;
  RealType* du;
  RealType* d2u;
  int first[2]; //We have up and down electrons.  This should be generalizd to nspecies.
  int last[2];

  TwoBodyJastrow(ParticleSet& p);
  TwoBodyJastrow(const TwoBodyJastrow& rhs) = default;
  ~TwoBodyJastrow();

  /* initialize storage */
  void init(ParticleSet& p);
  void initializeJastrowKokkos();
  
  /** add functor for (ia,ib) pair */
  void addFunc(int ia, int ib, FT* j);

  RealType evaluateLog(ParticleSet& P,
                       ParticleSet::ParticleGradient_t& G,
                       ParticleSet::ParticleLaplacian_t& L);

  /** recompute internal data assuming distance table is fully ready */
  void recompute(ParticleSet& P);

  ValueType ratio(ParticleSet& P, int iat);
  GradType evalGrad(ParticleSet& P, int iat);
  ValueType ratioGrad(ParticleSet& P, int iat, GradType& grad_iat);
  void acceptMove(ParticleSet& P, int iat);

  /** compute G and L after the sweep
   */
  void evaluateGL(ParticleSet& P,
                  ParticleSet::ParticleGradient_t& G,
                  ParticleSet::ParticleLaplacian_t& L,
                  bool fromscratch = false);

  /*@{ internal compute engines*/
  inline valT computeU(const ParticleSet& P, int iat, const RealType* restrict dist)
  {
    valT curUat(0);
    const int igt = P.GroupID[iat] * NumGroups;
    for (int jg = 0; jg < NumGroups; ++jg)
    {
      const FuncType& f2(F[igt + jg]);
      int iStart = P.first(jg);
      int iEnd   = P.last(jg);
      curUat += f2.evaluateV(iat, iStart, iEnd, dist, DistCompressed.data());
    }
    return curUat;
  }

  inline void computeU3(const ParticleSet& P,
                        int iat,
                        const RealType* restrict dist,
                        RealType* restrict u,
                        RealType* restrict du,
                        RealType* restrict d2u,
                        bool triangle = false);

  inline void operator()(const typename policy_t::member_type& team) const;

  /** compute gradient
   */
  inline posT accumulateG(const valT* restrict du, const RowContainer& displ) const
  {
    posT grad;
    for (int idim = 0; idim < OHMMS_DIM; ++idim)
    {
      const valT* restrict dX = displ.data(idim);
      valT s                  = valT();

      for (int jat = 0; jat < N; ++jat)

        s += du[jat] * dX[jat];
      grad[idim] = s;
    }
    return grad;
  }
  /**@} */


  /////////// Helpers to populate collective data structures
  template<typename atbjType, typename apsdType, typename vectorType, typename vectorType2>
  void populateCollectiveViews(atbjType atbjd, apsdType apsd, 
			       vectorType& WFC_list, vectorType2& P_list) {
    auto atbjdMirror = Kokkos::create_mirror_view(atbjd);
    auto apsdMirror = Kokkos::create_mirror_view(apsd);
    
    for (int i = 0; i < WFC_list.size(); i++) {
      atbjdMirror(i) = static_cast<TwoBodyJastrow*>(WFC_list[i])->jasData;
      apsdMirror(i) = P_list[i]->psk;
    }
    Kokkos::deep_copy(atbjd, atbjdMirror);
    Kokkos::deep_copy(apsd, apsdMirror);
  }

  template<typename atbjType, typename apsdType, typename vectorType, typename vectorType2>
  void populateCollectiveViews(atbjType atbjd, apsdType apsd, vectorType& WFC_list, 
			       vectorType2& P_list, const std::vector<bool>& isAccepted) {
    auto atbjdMirror = Kokkos::create_mirror_view(atbjd);
    auto apsdMirror = Kokkos::create_mirror_view(apsd);
    
    int idx = 0;
    for (int i = 0; i < WFC_list.size(); i++) {
      if (isAccepted[i]) {
	atbjdMirror(idx) = static_cast<TwoBodyJastrow*>(WFC_list[i])->jasData;
	apsdMirror(idx) = P_list[i]->psk;
	idx++;
      }
    }
    Kokkos::deep_copy(atbjd, atbjdMirror);
    Kokkos::deep_copy(apsd, apsdMirror);
  }

  //////////////////multi_evaluate functions
  // note that G_list and L_list feel redundant, they are just elements of P_list
  // if we wanted to be really kind though, we could copy them back out so this would
  // be where the host could see the results of the calcualtion easily
  virtual void multi_evaluateLog(const std::vector<WaveFunctionComponent*>& WFC_list,
                                 const std::vector<ParticleSet*>& P_list,
                                 const std::vector<ParticleSet::ParticleGradient_t*>& G_list,
                                 const std::vector<ParticleSet::ParticleLaplacian_t*>& L_list,
                                 ParticleSet::ParticleValue_t& values);

  virtual void multi_evaluateLog(const std::vector<WaveFunctionComponent*>& WFC_list,
				 WaveFunctionKokkos& wfc,
				 Kokkos::View<ParticleSet::pskType*>& psk,
				 ParticleSet::ParticleValue_t& values);

  virtual void multi_evalGrad(const std::vector<WaveFunctionComponent*>& WFC_list,
			      const std::vector<ParticleSet*>& P_list,
                              int iat, std::vector<posT>& grad_now);

  virtual void multi_evalGrad(const std::vector<WaveFunctionComponent*>& WFC_list,
			      WaveFunctionKokkos& wfc,
			      Kokkos::View<ParticleSet::pskType*>& psk,
                              int iat, std::vector<posT>& grad_now);

  virtual void multi_ratioGrad(const std::vector<WaveFunctionComponent*>& WFC_list,
                               WaveFunctionKokkos& wfc,
                               Kokkos::View<ParticleSet::pskType*> psk,
                               int iel,
                               Kokkos::View<int*>& isValidMap, int numValid,
                               std::vector<ValueType>& ratios,
                               std::vector<PosType>& grad_new); 

  virtual void multi_ratioGrad(const std::vector<WaveFunctionComponent*>& WFC_list,
			       const std::vector<ParticleSet*>& P_list,
			       int iat, std::vector<valT>& ratios,
			       std::vector<posT>& grad_new);
    
  virtual void multi_acceptRestoreMove(const std::vector<WaveFunctionComponent*>& WFC_list,
				       const std::vector<ParticleSet*>& P_list,
				       const std::vector<bool>& isAccepted,
				       int iat);

  virtual void multi_acceptrestoreMove(const std::vector<WaveFunctionComponent*>& WFC_list,
                                       WaveFunctionKokkos& wfc,
                                       Kokkos::View<ParticleSet::pskType*> psk,
                                       Kokkos::View<int*>& isAcceptedMap, int numAccepted, int iel);


  virtual void multi_evalRatio(int pairNum, Kokkos::View<int***>& eiList,
			       const std::vector<WaveFunctionComponent*>& WFC_list,
			       Kokkos::View<ParticleSetKokkos<RealType, ValueType, 3>*>& apsk,
			       Kokkos::View<double***>& likeTempR,
			       Kokkos::View<double***>& unlikeTempR,
			       Kokkos::View<int*>& activeWalkerIdx,
			       std::vector<ValueType>& ratios);
  
  virtual void multi_evaluateGL(const std::vector<WaveFunctionComponent*>& WFC_list,
				const std::vector<ParticleSet*>& P_list,
				const std::vector<ParticleSet::ParticleGradient_t*>& G_list,
				const std::vector<ParticleSet::ParticleLaplacian_t*>& L_list,
				bool fromscratch = false);


};

template<typename FT>
TwoBodyJastrow<FT>::TwoBodyJastrow(ParticleSet& p)
{
  init(p);
  FirstTime                 = true;
  KEcorr                    = 0.0;
  WaveFunctionComponentName = "TwoBodyJastrow";
}

template<typename FT>
TwoBodyJastrow<FT>::~TwoBodyJastrow()
{
//  auto it = J2Unique.begin();
//  while (it != J2Unique.end())
//  {
//    delete ((*it).second);
//    ++it;
 // }
} // need to clean up J2Unique

template<typename FT>
void TwoBodyJastrow<FT>::init(ParticleSet& p)
{
  N         = p.getTotalNum();
  N_padded  = getAlignedSize<valT>(N);
  NumGroups = p.groups();

  Uat.resize(N);
  dUat.resize(N);
  d2Uat.resize(N);

  //And now the Kokkos vectors
  cur_u   = Kokkos::View<valT*>("cur_u",N);
  cur_du  = Kokkos::View<valT*>("cur_du",N);
  cur_d2u = Kokkos::View<valT*>("cur_d2u",N);
  old_u   = Kokkos::View<valT*>("old_u",N);
  old_du  = Kokkos::View<valT*>("old_du",N);
  old_d2u = Kokkos::View<valT*>("old_d2u",N);
  DistIndice=Kokkos::View<int*>("DistIndice",N);
  DistCompressed=Kokkos::View<valT*>("DistCompressed",N);

  F = Kokkos::View<FT*,F_device_type>("FT",NumGroups * NumGroups);
  for(int i=0; i<NumGroups*NumGroups; i++){
    new(&F(i)) FT();
  }
  initializeJastrowKokkos();
}

template<typename FT>
void TwoBodyJastrow<FT>::initializeJastrowKokkos() {
  jasData.LogValue       = Kokkos::View<valT[1]>("LogValue");

  jasData.Nelec          = Kokkos::View<int[1]>("Nelec");
  auto NelecMirror       = Kokkos::create_mirror_view(jasData.Nelec);
  NelecMirror(0)         = N;
  Kokkos::deep_copy(jasData.Nelec, NelecMirror);

  jasData.NumGroups      = Kokkos::View<int[1]>("NumGroups");
  auto NumGroupsMirror   = Kokkos::create_mirror_view(jasData.NumGroups);
  NumGroupsMirror(0)     = NumGroups;
  Kokkos::deep_copy(jasData.NumGroups, NumGroupsMirror);

  jasData.first          = Kokkos::View<int[2]>("first");
  auto firstMirror       = Kokkos::create_mirror_view(jasData.first);
  firstMirror(0)         = first[0];
  firstMirror(1)         = first[1];
  Kokkos::deep_copy(jasData.first, firstMirror);

  jasData.last           = Kokkos::View<int[2]>("last");
  auto lastMirror        = Kokkos::create_mirror_view(jasData.last);
  lastMirror(0)          = last[0];
  lastMirror(1)          = last[1];
  Kokkos::deep_copy(jasData.last, lastMirror);

  jasData.updateMode      = Kokkos::View<int[1]>("updateMode");
  auto updateModeMirror   = Kokkos::create_mirror_view(jasData.updateMode);
  updateModeMirror(0)     = 3;
  Kokkos::deep_copy(jasData.updateMode, updateModeMirror);

  jasData.cur_Uat         = Kokkos::View<valT[1]>("cur_Uat");
  auto cur_UatMirror      = Kokkos::create_mirror_view(jasData.cur_Uat);
  jasData.Uat             = Kokkos::View<valT*>("Uat", N);
  auto UatMirror          = Kokkos::create_mirror_view(jasData.Uat);
  jasData.dUat            = Kokkos::View<valT*[OHMMS_DIM], Kokkos::LayoutLeft>("dUat", N);
  auto dUatMirror         = Kokkos::create_mirror_view(jasData.dUat);
  jasData.d2Uat           = Kokkos::View<valT*>("d2Uat", N);
  auto d2UatMirror        = Kokkos::create_mirror_view(jasData.d2Uat);

  cur_UatMirror(0)        = cur_Uat;
  for (int i = 0; i < N; i++) {
    UatMirror(i) = Uat[i];
    d2UatMirror(i) = d2Uat[i];
    for (int j = 0; j < OHMMS_DIM; j++) {
      dUatMirror(i,j) = dUat[i][j];
    }
  }
  Kokkos::deep_copy(jasData.Uat, UatMirror);
  Kokkos::deep_copy(jasData.dUat, dUatMirror);
  Kokkos::deep_copy(jasData.d2Uat, d2UatMirror);
  Kokkos::deep_copy(jasData.cur_Uat, cur_UatMirror);

  // these things are already views, so just do operator=
  jasData.cur_u = cur_u;
  jasData.old_u = old_u;
  jasData.cur_du = cur_du;
  jasData.old_du = old_du;
  jasData.cur_d2u = cur_d2u;
  jasData.old_d2u = old_d2u;
  jasData.DistCompressed = DistCompressed;
  jasData.DistIndices = DistIndice;

  // need to put in the data for A, dA and d2A    
  TinyVector<valT, 16> A(-1.0/6.0,  3.0/6.0, -3.0/6.0, 1.0/6.0,
  			  3.0/6.0, -6.0/6.0,  0.0/6.0, 4.0/6.0,
			 -3.0/6.0,  3.0/6.0,  3.0/6.0, 1.0/6.0,
			  1.0/6.0,  0.0/6.0,  0.0/6.0, 0.0/6.0);
  TinyVector<valT,16>  dA(0.0, -0.5,  1.0, -0.5,
			  0.0,  1.5, -2.0,  0.0,
			  0.0, -1.5,  1.0,  0.5,
			  0.0,  0.5,  0.0,  0.0);
  TinyVector<valT,16>  d2A(0.0, 0.0, -1.0,  1.0,
			   0.0, 0.0,  3.0, -2.0,
			   0.0, 0.0, -3.0,  1.0,
			   0.0, 0.0,  1.0,  0.0);
  
  jasData.A              = Kokkos::View<valT[16]>("A");
  auto Amirror           = Kokkos::create_mirror_view(jasData.A);
  jasData.dA             = Kokkos::View<valT[16]>("dA");
  auto dAmirror          = Kokkos::create_mirror_view(jasData.dA);
  jasData.d2A            = Kokkos::View<valT[16]>("d2A");
  auto d2Amirror         = Kokkos::create_mirror_view(jasData.d2A);

  for (int i = 0; i < 16; i++) {
    Amirror(i) = A[i];
    dAmirror(i) = dA[i];
    d2Amirror(i) = d2A[i];
  }

  Kokkos::deep_copy(jasData.A, Amirror);
  Kokkos::deep_copy(jasData.dA, dAmirror);
  Kokkos::deep_copy(jasData.d2A, d2Amirror);

  // also set up and allocate memory for cutoff_radius, DeltaRInv
  jasData.cutoff_radius   = Kokkos::View<valT*>("Cutoff_Radii", NumGroups*NumGroups);
  jasData.DeltaRInv       = Kokkos::View<valT*>("DeltaRInv", NumGroups*NumGroups);
  
  // unfortunately have to defer setting up SplineCoefs because we don't yet know
  // how many elements are in SplineCoefs on the cpu
  splCoefsNotAllocated   = true;
}
  
template<typename FT>
void TwoBodyJastrow<FT>::evaluateGL(ParticleSet& P,
                                    ParticleSet::ParticleGradient_t& G,
                                    ParticleSet::ParticleLaplacian_t& L,
                                    bool fromscratch)
{
  if (fromscratch)
    recompute(P);
  LogValue = valT(0);
  for (int iat = 0; iat < N; ++iat)
  {
    LogValue += Uat[iat];
    G[iat] += dUat[iat];
    L[iat] += d2Uat[iat];
  }

  constexpr valT mhalf(-0.5);
  LogValue = mhalf * LogValue;
}


template<typename FT>
void TwoBodyJastrow<FT>::addFunc(int ia, int ib, FT* j)
{
  if (splCoefsNotAllocated) {
    splCoefsNotAllocated = false;
    jasData.SplineCoefs   = Kokkos::View<valT**>("SplineCoefficients", NumGroups*NumGroups, j->SplineCoefs.extent(0));
  }

  if (ia == ib)
  {
    if (ia == 0) // first time, assign everything
    {
      int ij = 0;
      for (int ig = 0; ig < NumGroups; ++ig)
        for (int jg = 0; jg < NumGroups; ++jg, ++ij) {
	  F[ij] = *j;
	  auto crMirror = Kokkos::create_mirror_view(jasData.cutoff_radius);
	  auto drinvMirror = Kokkos::create_mirror_view(jasData.cutoff_radius);
	  Kokkos::deep_copy(crMirror, jasData.cutoff_radius);
	  Kokkos::deep_copy(drinvMirror, jasData.DeltaRInv);
	  crMirror(ij) = j->cutoff_radius;
	  drinvMirror(ij) = j->DeltaRInv;
	  Kokkos::deep_copy(jasData.cutoff_radius, crMirror);
	  Kokkos::deep_copy(jasData.DeltaRInv, drinvMirror);
	  
	  auto bigScMirror   = Kokkos::create_mirror_view(jasData.SplineCoefs);
	  auto smallScMirror = Kokkos::create_mirror_view(j->SplineCoefs);
	  Kokkos::deep_copy(smallScMirror, j->SplineCoefs);
	  Kokkos::deep_copy(bigScMirror,   jasData.SplineCoefs);
	  for (int i = 0; i < j->SplineCoefs.extent(0); i++) {
	    bigScMirror(ij, i) = smallScMirror(i);
	  }
	  Kokkos::deep_copy(jasData.SplineCoefs, bigScMirror);
	}
    }
    else {
      F[ia * NumGroups + ib] = *j;
      int groupIndex = ia*NumGroups + ib;
      auto crMirror = Kokkos::create_mirror_view(jasData.cutoff_radius);
      auto drinvMirror = Kokkos::create_mirror_view(jasData.cutoff_radius);
      Kokkos::deep_copy(crMirror, jasData.cutoff_radius);
      Kokkos::deep_copy(drinvMirror, jasData.DeltaRInv);
      crMirror(groupIndex) = j->cutoff_radius;
      drinvMirror(groupIndex) = j->DeltaRInv;
      Kokkos::deep_copy(jasData.cutoff_radius, crMirror);
      Kokkos::deep_copy(jasData.DeltaRInv, drinvMirror);
      
      auto bigScMirror   = Kokkos::create_mirror_view(jasData.SplineCoefs);
      auto smallScMirror = Kokkos::create_mirror_view(j->SplineCoefs);
      Kokkos::deep_copy(smallScMirror, j->SplineCoefs);
      Kokkos::deep_copy(bigScMirror,   jasData.SplineCoefs);
      for (int i = 0; i < j->SplineCoefs.extent(0); i++) {
	bigScMirror(groupIndex, i) = smallScMirror(i);
      }
      Kokkos::deep_copy(jasData.SplineCoefs, bigScMirror);
    }
    
  }
  else
  {
    if (N == 2)
    {
      // a very special case, 1 up + 1 down
      // uu/dd was prevented by the builder
      for (int ig = 0; ig < NumGroups; ++ig)
        for (int jg = 0; jg < NumGroups; ++jg) {
          F[ig * NumGroups + jg] = *j;
	  int groupIndex = ig*NumGroups + jg;
	  auto crMirror = Kokkos::create_mirror_view(jasData.cutoff_radius);
	  auto drinvMirror = Kokkos::create_mirror_view(jasData.cutoff_radius);
	  Kokkos::deep_copy(crMirror, jasData.cutoff_radius);
	  Kokkos::deep_copy(drinvMirror, jasData.DeltaRInv);
	  crMirror(groupIndex) = j->cutoff_radius;
	  drinvMirror(groupIndex) = j->DeltaRInv;
	  Kokkos::deep_copy(jasData.cutoff_radius, crMirror);
	  Kokkos::deep_copy(jasData.DeltaRInv, drinvMirror);
	  
	  auto bigScMirror   = Kokkos::create_mirror_view(jasData.SplineCoefs);
	  auto smallScMirror = Kokkos::create_mirror_view(j->SplineCoefs);
	  Kokkos::deep_copy(smallScMirror, j->SplineCoefs);
	  Kokkos::deep_copy(bigScMirror,   jasData.SplineCoefs);
	  for (int i = 0; i < j->SplineCoefs.extent(0); i++) {
	    bigScMirror(groupIndex, i) = smallScMirror(i);
	  }
	  Kokkos::deep_copy(jasData.SplineCoefs, bigScMirror);
	}
    }
    else
    {
      // generic case
      F[ia * NumGroups + ib] = *j;
      int groupIndex = ia*NumGroups + ib;
      auto crMirror = Kokkos::create_mirror_view(jasData.cutoff_radius);
      auto drinvMirror = Kokkos::create_mirror_view(jasData.DeltaRInv);
      Kokkos::deep_copy(crMirror, jasData.cutoff_radius);
      Kokkos::deep_copy(drinvMirror, jasData.DeltaRInv);
      crMirror(groupIndex) = j->cutoff_radius;
      drinvMirror(groupIndex) = j->DeltaRInv;
      Kokkos::deep_copy(jasData.cutoff_radius, crMirror);
      Kokkos::deep_copy(jasData.DeltaRInv, drinvMirror);
      
      auto bigScMirror   = Kokkos::create_mirror_view(jasData.SplineCoefs);
      auto smallScMirror = Kokkos::create_mirror_view(j->SplineCoefs);
      Kokkos::deep_copy(smallScMirror, j->SplineCoefs);
      Kokkos::deep_copy(bigScMirror,   jasData.SplineCoefs);
      for (int i = 0; i < j->SplineCoefs.extent(0); i++) {
	bigScMirror(groupIndex, i) = smallScMirror(i);
      }
      Kokkos::deep_copy(jasData.SplineCoefs, bigScMirror);
    }
  }
  std::stringstream aname;
  aname << ia << ib;
//  J2Unique[aname.str()] = *j;
  FirstTime             = false;
}

/** intenal function to compute \f$\sum_j u(r_j), du/dr, d2u/dr2\f$
 * @param P particleset
 * @param iat particle index
 * @param dist starting distance
 * @param u starting value
 * @param du starting first deriv
 * @param d2u starting second deriv
 */
template<typename FT>
inline void TwoBodyJastrow<FT>::computeU3(const ParticleSet& P,
                                          int iat_,
                                          const RealType* restrict dist_,
                                          RealType* restrict u_,
                                          RealType* restrict du_,
                                          RealType* restrict d2u_,
                                          bool triangle)
{
  iat = iat_;
  dist = dist_;
  u = u_;
  du = du_;
  d2u = d2u_;

  const int jelmax = triangle ? iat : N;
  constexpr valT czero(0);
  std::fill_n(u, jelmax, czero);
  std::fill_n(du, jelmax, czero);
  std::fill_n(d2u, jelmax, czero);

  igt = P.GroupID[iat] * NumGroups;
  for (int jg = 0; jg < NumGroups; ++jg)
  {
    const FuncType& f2(F[igt + jg]);
    jg_hack = jg;
    first[jg]  = P.first(jg);
    last[jg]   = std::min(jelmax, P.last(jg));
   // f2.evaluateVGL(iat, iStart, iEnd, dist, u, du, d2u, DistCompressed.data(), DistIndice.data());
    Kokkos::parallel_for(policy_t(1,1,32),*this);
  }
  // u[iat]=czero;
  // du[iat]=czero;
  // d2u[iat]=czero;
}

template<typename FT>
KOKKOS_INLINE_FUNCTION void TwoBodyJastrow<FT>::operator() (const typename policy_t::member_type& team) const {
  int jg = jg_hack;
  int iStart = first[jg];
  int iEnd = last[jg];
 // printf("Hi %d %d %d\n",jg,iStart,iEnd);
  F[igt+jg].evaluateVGL(team,iat,iStart, iEnd, dist, u, du, d2u, DistCompressed.data(),
                        DistIndice.data());
}

template<typename FT>
typename TwoBodyJastrow<FT>::ValueType TwoBodyJastrow<FT>::ratio(ParticleSet& P, int iat)
{
  // only ratio, ready to compute it again
  UpdateMode = ORB_PBYP_RATIO;
  cur_Uat    = computeU(P, iat, P.DistTables[0]->Temp_r.data());
  return std::exp(Uat[iat] - cur_Uat);
}

template<typename FT>
typename TwoBodyJastrow<FT>::GradType TwoBodyJastrow<FT>::evalGrad(ParticleSet& P, int iat)
{
  return GradType(dUat[iat]);
}

template<typename FT>
typename TwoBodyJastrow<FT>::ValueType
    TwoBodyJastrow<FT>::ratioGrad(ParticleSet& P, int iat, GradType& grad_iat)
{
  UpdateMode = ORB_PBYP_PARTIAL;

  computeU3(P, iat, P.DistTables[0]->Temp_r.data(), cur_u.data(), cur_du.data(), cur_d2u.data());
  cur_Uat = simd::accumulate_n(cur_u.data(), N, valT());
  DiffVal = Uat[iat] - cur_Uat;
  grad_iat += accumulateG(cur_du.data(), P.DistTables[0]->Temp_dr);
  return std::exp(DiffVal);
}

template<typename FT>
void TwoBodyJastrow<FT>::acceptMove(ParticleSet& P, int iat)
{
  // get the old u, du, d2u
  const DistanceTableData* d_table = P.DistTables[0];
  computeU3(P, iat, d_table->Distances[iat], old_u.data(), old_du.data(), old_d2u.data());
  if (UpdateMode == ORB_PBYP_RATIO)
  { // ratio-only during the move; need to compute derivatives
    const auto dist = d_table->Temp_r.data();
    computeU3(P, iat, dist, cur_u.data(), cur_du.data(), cur_d2u.data());
  }

  valT cur_d2Uat(0);
  const auto& new_dr    = d_table->Temp_dr;
  const auto& old_dr    = d_table->Displacements[iat];
  constexpr valT lapfac = OHMMS_DIM - RealType(1);
  for (int jat = 0; jat < N; jat++)
  {
    const valT du   = cur_u[jat] - old_u[jat];
    const valT newl = cur_d2u[jat] + lapfac * cur_du[jat];
    const valT dl   = old_d2u[jat] + lapfac * old_du[jat] - newl;
    Uat[jat] += du;
    d2Uat[jat] += dl;
    cur_d2Uat -= newl;
  }
  posT cur_dUat;
  for (int idim = 0; idim < OHMMS_DIM; ++idim)
  {
    const valT* restrict new_dX    = new_dr.data(idim);
    const valT* restrict old_dX    = old_dr.data(idim);
    const valT* restrict cur_du_pt = cur_du.data();
    const valT* restrict old_du_pt = old_du.data();
    valT* restrict save_g          = dUat.data(idim);
    valT cur_g                     = cur_dUat[idim];
    for (int jat = 0; jat < N; jat++)
    {
      const valT newg = cur_du_pt[jat] * new_dX[jat];
      const valT dg   = newg - old_du_pt[jat] * old_dX[jat];
      save_g[jat] -= dg;
      cur_g += newg;
    }
    cur_dUat[idim] = cur_g;
  }
  LogValue += Uat[iat] - cur_Uat;
  Uat[iat]   = cur_Uat;
  dUat(iat)  = cur_dUat;
  d2Uat[iat] = cur_d2Uat;
}

template<typename FT>
void TwoBodyJastrow<FT>::recompute(ParticleSet& P)
{
  const DistanceTableData* d_table = P.DistTables[0];
  for (int ig = 0; ig < NumGroups; ++ig)
  {
    const int igt = ig * NumGroups;
    for (int iat = P.first(ig), last = P.last(ig); iat < last; ++iat)
    {
      computeU3(P, iat, d_table->Distances[iat], cur_u.data(), cur_du.data(), cur_d2u.data(), true);
      Uat[iat] = simd::accumulate_n(cur_u.data(), iat, valT());
      posT grad;
      valT lap(0);
      const valT* restrict u    = cur_u.data();
      const valT* restrict du   = cur_du.data();
      const valT* restrict d2u  = cur_d2u.data();
      const RowContainer& displ = d_table->Displacements[iat];
      constexpr valT lapfac     = OHMMS_DIM - RealType(1);
      for (int jat = 0; jat < iat; ++jat)
        lap += d2u[jat] + lapfac * du[jat];
      for (int idim = 0; idim < OHMMS_DIM; ++idim)
      {
        const valT* restrict dX = displ.data(idim);
        valT s                  = valT();
        for (int jat = 0; jat < iat; ++jat)
          s += du[jat] * dX[jat];
        grad[idim] = s;
      }
      dUat(iat)  = grad;
      d2Uat[iat] = -lap;
      // add the contribution from the upper triangle
      for (int jat = 0; jat < iat; jat++)
      {
        Uat[jat] += u[jat];
        d2Uat[jat] -= d2u[jat] + lapfac * du[jat];
      }
      for (int idim = 0; idim < OHMMS_DIM; ++idim)
      {
        valT* restrict save_g   = dUat.data(idim);
        const valT* restrict dX = displ.data(idim);
        for (int jat = 0; jat < iat; jat++)
          save_g[jat] -= du[jat] * dX[jat];
      }
    }
  }
}

template<typename FT>
typename TwoBodyJastrow<FT>::RealType
    TwoBodyJastrow<FT>::evaluateLog(ParticleSet& P,
                                    ParticleSet::ParticleGradient_t& G,
                                    ParticleSet::ParticleLaplacian_t& L)
{
  evaluateGL(P, G, L, true);
  return LogValue;
}

template<typename FT>
void TwoBodyJastrow<FT>::multi_evaluateLog(const std::vector<WaveFunctionComponent*>& WFC_list,
					   const std::vector<ParticleSet*>& P_list,
					   const std::vector<ParticleSet::ParticleGradient_t*>& G_list,
					   const std::vector<ParticleSet::ParticleLaplacian_t*>& L_list,
					   ParticleSet::ParticleValue_t& values) {
  // make a view of all of the TwoBodyJastrowData and relevantParticleSetData
  Kokkos::View<jasDataType*> allTwoBodyJastrowData("atbjd", WFC_list.size()); 
  Kokkos::View<ParticleSet::pskType*> allParticleSetData("apsd", P_list.size());
  populateCollectiveViews(allTwoBodyJastrowData, allParticleSetData, WFC_list, P_list);
  
  // need to make a view to hold all of the output LogValues
  Kokkos::View<ValueType*> tempValues("tempValues", P_list.size());
  
  // need to write this function
  doTwoBodyJastrowMultiEvaluateLog(allTwoBodyJastrowData, allParticleSetData, tempValues);
  
  // copy the results out to values
  auto tempValMirror = Kokkos::create_mirror_view(tempValues);
  Kokkos::deep_copy(tempValMirror, tempValues);
  
  for (int i = 0; i < P_list.size(); i++) {
    values[i] = tempValMirror(i);
  }
}

template<typename FT>
void TwoBodyJastrow<FT>::multi_evaluateLog(const std::vector<WaveFunctionComponent*>& WFC_list,
					   WaveFunctionKokkos& wfc,
					   Kokkos::View<ParticleSet::pskType*>& psk,
					   ParticleSet::ParticleValue_t& values) {
  
  // need to write this function
  doTwoBodyJastrowMultiEvaluateLog(wfc.twoBodyJastrows, psk, wfc.ratios_view);

  Kokkos::deep_copy(wfc.ratios_view_mirror, wfc.ratios_view);
  
  for (int i = 0; i < WFC_list.size(); i++) {
    values[i] = wfc.ratios_view_mirror(i);
  }
}
 
template<typename FT>
void TwoBodyJastrow<FT>::multi_evalGrad(const std::vector<WaveFunctionComponent*>& WFC_list,
					const std::vector<ParticleSet*>& P_list,
					int iat, std::vector<posT>& grad_now) {
  // make a view of all of the TwoBodyJastrowData and relevantParticleSetData
  Kokkos::View<jasDataType*> allTwoBodyJastrowData("atbjd", WFC_list.size()); 
  Kokkos::View<ParticleSet::pskType*> allParticleSetData("apsd", P_list.size());
  populateCollectiveViews(allTwoBodyJastrowData, allParticleSetData, WFC_list, P_list);
  
  // need to make a view to hold all of the output LogValues
  Kokkos::View<double**> grad_now_view("tempValues", P_list.size(), OHMMS_DIM);
  
  // need to write this function
  doTwoBodyJastrowMultiEvalGrad(allTwoBodyJastrowData, iat, grad_now_view);
  
  // copy the results out to values
  auto grad_now_view_mirror = Kokkos::create_mirror_view(grad_now_view);
  Kokkos::deep_copy(grad_now_view_mirror, grad_now_view);
  
  for (int i = 0; i < P_list.size(); i++) {
    for (int j = 0; j < OHMMS_DIM; j++) {
      grad_now[i][j] = grad_now_view_mirror(i,j);
    }
  }
}

template<typename FT>
void TwoBodyJastrow<FT>::multi_evalGrad(const std::vector<WaveFunctionComponent*>& WFC_list,
					WaveFunctionKokkos& wfc,
					Kokkos::View<ParticleSet::pskType*>& psk,
					int iat,
					std::vector<posT>& grad_now) {
  const int numItems = WFC_list.size();

  // need to write this function
  doTwoBodyJastrowMultiEvalGrad(wfc.twoBodyJastrows, iat, wfc.grad_view);
  // copy the results out to values
  Kokkos::deep_copy(wfc.grad_view_mirror, wfc.grad_view);

  for (int i = 0; i < numItems; i++) {
    for (int j = 0; j < OHMMS_DIM; j++) {
      grad_now[i][j] = wfc.grad_view_mirror(i,j);
    }
  }
}

template<typename FT>
void TwoBodyJastrow<FT>::multi_ratioGrad(const std::vector<WaveFunctionComponent*>& WFC_list,
					 WaveFunctionKokkos& wfc,
					 Kokkos::View<ParticleSet::pskType*> psk,
					 int iel,
					 Kokkos::View<int*>& isValidMap, int numValid,
					 std::vector<ValueType>& ratios,
					 std::vector<PosType>& grad_new) {
  if (numValid > 0) {

    doTwoBodyJastrowMultiRatioGrad(wfc.twoBodyJastrows, psk, isValidMap, numValid, iel, wfc.grad_view, wfc.ratios_view);
    Kokkos::fence();

    // copy the results out to values                                                                                                          
    Kokkos::deep_copy(wfc.grad_view_mirror, wfc.grad_view);
    Kokkos::deep_copy(wfc.ratios_view_mirror, wfc.ratios_view);
    //std::cout << "       finished copying grad and ratios out" << std::endl;                                                                 

    for (int i = 0; i < numValid; i++) {
      ratios[i] = wfc.ratios_view_mirror(i);
      for (int j = 0; j < OHMMS_DIM; j++) {
	grad_new[i][j] += wfc.grad_view_mirror(i,j);
      }
    }
  }
  //std::cout << "      finishing J1 multi_ratioGrad" << std::endl;                                                                            
 }



template<typename FT>
void TwoBodyJastrow<FT>::multi_ratioGrad(const std::vector<WaveFunctionComponent*>& WFC_list,
					 const std::vector<ParticleSet*>& P_list,
					 int iat, std::vector<valT>& ratios,
					 std::vector<posT>& grad_new) {
  // make a view of all of the TwoBodyJastrowData and relevantParticleSetData
  Kokkos::View<jasDataType*> allTwoBodyJastrowData("atbjd", WFC_list.size()); 
  Kokkos::View<ParticleSet::pskType*> allParticleSetData("apsd", P_list.size());
  populateCollectiveViews(allTwoBodyJastrowData, allParticleSetData, WFC_list, P_list);
  
  // need to make a view to hold all of the output LogValues
  Kokkos::View<double**> grad_new_view("tempValues", P_list.size(), OHMMS_DIM);
  Kokkos::View<double*> ratios_view("ratios", P_list.size());
    
  // need to write this function
  doTwoBodyJastrowMultiRatioGrad(allTwoBodyJastrowData, allParticleSetData, iat, grad_new_view, ratios_view);

  // copy the results out to values
  auto grad_new_view_mirror = Kokkos::create_mirror_view(grad_new_view);
  Kokkos::deep_copy(grad_new_view_mirror, grad_new_view);
  auto ratios_view_mirror = Kokkos::create_mirror_view(ratios_view);
  Kokkos::deep_copy(ratios_view_mirror, ratios_view);
  
  for (int i = 0; i < P_list.size(); i++) {
    ratios[i] = ratios_view_mirror(i);
    for (int j = 0; j < OHMMS_DIM; j++) {
      grad_new[i][j] += grad_new_view_mirror(i,j);
    }
  }
}

// still need to fix this one
template<typename FT>
void TwoBodyJastrow<FT>::multi_evalRatio(int pairNum, Kokkos::View<int***>& eiList,
					 const std::vector<WaveFunctionComponent*>& WFC_list,
					 Kokkos::View<ParticleSetKokkos<RealType, ValueType, 3>*>& apsk,
					 Kokkos::View<double***>& likeTempR,
					 Kokkos::View<double***>& unlikeTempR,
					 Kokkos::View<int*>& activeWalkerIdx,
					 std::vector<ValueType>& ratios) {
  const int numActiveWalkers = activeWalkerIdx.extent(0);
  const int numKnots = likeTempR.extent(1);
  
  auto activeWalkerIdxMirror = Kokkos::create_mirror_view(activeWalkerIdx);
  Kokkos::deep_copy(activeWalkerIdxMirror, activeWalkerIdx);
  
  Kokkos::View<jasDataType*> allTwoBodyJastrowData("atbjd", activeWalkerIdx.extent(0));
  auto atbjdMirror = Kokkos::create_mirror_view(allTwoBodyJastrowData);
  for (int i = 0; i < numActiveWalkers; i++) {
    //const int walkerIdx = activeWalkerIdxMirror(i);
    const int walkerIdx = i;
    atbjdMirror(i) = static_cast<TwoBodyJastrow*>(WFC_list[walkerIdx])->jasData;
  }
  Kokkos::deep_copy(allTwoBodyJastrowData, atbjdMirror);
  
  Kokkos::View<ValueType**> devRatios("tbjDevRatios", numActiveWalkers, numKnots);
  
  doTwoBodyJastrowMultiEvalRatio(pairNum, eiList, apsk, allTwoBodyJastrowData, likeTempR, activeWalkerIdx, devRatios);
  
  auto devRatiosMirror = Kokkos::create_mirror_view(devRatios);
  Kokkos::deep_copy(devRatiosMirror, devRatios);
  for (int i = 0; i < devRatiosMirror.extent(0); i++) {
    const int walkerIndex = activeWalkerIdxMirror(i);
    for (int j = 0; j < devRatiosMirror.extent(1); j++) {
      ratios[walkerIndex*numKnots+j] = devRatiosMirror(i,j);
    }
  }
}
template<typename FT>
void TwoBodyJastrow<FT>::multi_acceptrestoreMove(const std::vector<WaveFunctionComponent*>& WFC_list,
						 WaveFunctionKokkos& wfc,
						 Kokkos::View<ParticleSet::pskType*> psk,
						 Kokkos::View<int*>& isAcceptedMap, int numAccepted, int iel) {
  doTwoBodyJastrowMultiAcceptRestoreMove(wfc.twoBodyJastrows, psk, isAcceptedMap, numAccepted, iel);
}


template<typename FT>
void TwoBodyJastrow<FT>::multi_acceptRestoreMove(const std::vector<WaveFunctionComponent*>& WFC_list,
						 const std::vector<ParticleSet*>& P_list,
						 const std::vector<bool>& isAccepted, int iat) {
  int numAccepted = 0;
  for (int i = 0; i < isAccepted.size(); i++) {
    if (isAccepted[i]) {
      numAccepted++;
    }
  }
  
  // make a view of all of the TwoBodyJastrowData and relevantParticleSetData
  Kokkos::View<jasDataType*> allTwoBodyJastrowData("atbjd", numAccepted); 
  Kokkos::View<ParticleSet::pskType*> allParticleSetData("apsd", numAccepted);
  populateCollectiveViews(allTwoBodyJastrowData, allParticleSetData, WFC_list, P_list, isAccepted);
  
  // need to write this function
  doTwoBodyJastrowMultiAcceptRestoreMove(allTwoBodyJastrowData, allParticleSetData, iat);
  
  // be careful on this one, looks like it is being done for side effects.  Should see what needs to go back!!!
}

template<typename FT>
void TwoBodyJastrow<FT>::multi_evaluateGL(const std::vector<WaveFunctionComponent*>& WFC_list,
					  const std::vector<ParticleSet*>& P_list,
					  const std::vector<ParticleSet::ParticleGradient_t*>& G_list,
					  const std::vector<ParticleSet::ParticleLaplacian_t*>& L_list,
					  bool fromscratch) {
  // make a view of all of the TwoBodyJastrowData and relevantParticleSetData
  Kokkos::View<jasDataType*> allTwoBodyJastrowData("atbjd", WFC_list.size()); 
  Kokkos::View<ParticleSet::pskType*> allParticleSetData("apsd", P_list.size());
  populateCollectiveViews(allTwoBodyJastrowData, allParticleSetData, WFC_list, P_list);
  
  // need to write this function
  doTwoBodyJastrowMultiEvaluateGL(allTwoBodyJastrowData, allParticleSetData, fromscratch);
  
  // know that we will need LogValue to up updated after this, possibly other things in ParticleSet!!!
  for (int i = 0; i < WFC_list.size(); i++) {
    auto LogValueMirror = Kokkos::create_mirror_view(static_cast<TwoBodyJastrow*>(WFC_list[i])->jasData.LogValue);
    Kokkos::deep_copy(LogValueMirror, static_cast<TwoBodyJastrow*>(WFC_list[i])->jasData.LogValue);
    LogValue = LogValueMirror(0);
  }
}
 

} // namespace qmcplusplus
#endif
