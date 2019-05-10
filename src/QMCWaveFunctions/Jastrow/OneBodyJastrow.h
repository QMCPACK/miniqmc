//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License. See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by:
//
// File created by: Jeongnim Kim, jeongnim.kim@intel.com, Intel Corp.
//////////////////////////////////////////////////////////////////////////////////////
// -*- C++ -*-
#ifndef QMCPLUSPLUS_ONEBODYJASTROW_H
#define QMCPLUSPLUS_ONEBODYJASTROW_H
#include "Utilities/Configuration.h"
#include "QMCWaveFunctions/WaveFunctionComponent.h"
#include <Utilities/SIMD/allocator.hpp>
#include <Utilities/SIMD/algorithm.hpp>
#include <numeric>
#include "OneBodyJastrowKokkos.h"
#include "QMCWaveFunctions/WaveFunctionKokkos.h"

/*!
 * @file OneBodyJastrow.h
 */

namespace qmcplusplus
{

template<typename aobjdType, typename apsdType>
void doOneBodyJastrowMultiEvaluateGL(aobjdType aobjd, apsdType apsd, int nelec, bool fromscratch) {
  const int numWalkers = aobjd.extent(0);
  using BarePolicy = Kokkos::TeamPolicy<>;
  BarePolicy pol(numWalkers*nelec, 1, 32);
  Kokkos::parallel_for("obj-evalGL-waker-loop", pol,
		       KOKKOS_LAMBDA(BarePolicy::member_type member) {
			 int walkerNum = member.league_rank()/nelec; 
			 aobjd(walkerNum).evaluateGL(member, apsd(walkerNum), fromscratch);
		       });
}

template<typename aobjdType, typename apsdType>
void doOneBodyJastrowMultiAcceptRestoreMove(aobjdType aobjd, apsdType apsd,
					    Kokkos::View<int*>& isAcceptedMap,
					    int numAccepted,int iat) {
  const int numWalkers = numAccepted;
  using BarePolicy = Kokkos::TeamPolicy<>;
  BarePolicy pol(numAccepted, 1, 32);
  Kokkos::parallel_for("obj-acceptRestoreMove-waker-loop", pol,
		       KOKKOS_LAMBDA(BarePolicy::member_type member) {
			 int walkerIdx = member.league_rank(); 
			 int walkerNum = isAcceptedMap(walkerIdx);
			 aobjd(walkerNum).acceptMove(member, apsd(walkerNum), iat);
		       });
}


template<typename aobjdType, typename apsdType>
void doOneBodyJastrowMultiAcceptRestoreMove(aobjdType aobjd, apsdType apsd, int iat) {
  const int numWalkers = aobjd.extent(0);
  using BarePolicy = Kokkos::TeamPolicy<>;
  BarePolicy pol(numWalkers, 1, 32);
  Kokkos::parallel_for("obj-acceptRestoreMove-waker-loop", pol,
		       KOKKOS_LAMBDA(BarePolicy::member_type member) {
			 int walkerNum = member.league_rank(); 
			 aobjd(walkerNum).acceptMove(member, apsd(walkerNum), iat);
		       });
}

template<typename aobjdType, typename apsdType, typename valT>
void doOneBodyJastrowMultiRatioGrad(aobjdType& aobjd, apsdType& apsd, Kokkos::View<int*>& isValidMap,
			       int numValid, int iel, Kokkos::View<valT**> gradNowView,
			       Kokkos::View<valT*> ratiosView) {
  const int numWalkers = numValid;
  const int numIons = aobjd(0).Nions(0); // note this is bad, relies on UVM, kill it
  
  Kokkos::Profiling::pushRegion("obj-evalRatioGrad");
  Kokkos::parallel_for("obj-evalRatioGrad-part1",
		       Kokkos::RangePolicy<>(0,numWalkers*numIons),
		       KOKKOS_LAMBDA(const int &idx) {
			 const int walkerIdx = idx / numIons;
			 const int walkerNum = isValidMap(walkerIdx);
			 const int workingIonNum = idx % numIons;
			 aobjd(walkerNum).ratioGrad_part1(apsd(walkerNum), workingIonNum);
		       });
  Kokkos::parallel_for("obj-evalRatioGrad-part2",
		       Kokkos::RangePolicy<>(0,numWalkers*numIons),
		       KOKKOS_LAMBDA(const int &idx) {
			 const int walkerIdx = idx / numIons;
			 const int walkerNum = isValidMap(walkerIdx);
			 const int workingIonNum = idx % numIons;
			 aobjd(walkerNum).ratioGrad_part2(apsd(walkerNum), workingIonNum);
		       });
  Kokkos::parallel_for("obj-evalRatioGrad-part3",
		       Kokkos::RangePolicy<>(0,numWalkers),
		       KOKKOS_LAMBDA(const int &idx) {
			 const int walkerIdx = idx;
			 const int walkerNum = isValidMap(walkerIdx);
			 auto gv = Kokkos::subview(gradNowView,walkerIdx,Kokkos::ALL());
			 ratiosView(walkerIdx) = aobjd(walkerNum).ratioGrad_part3(iel, gv);
		       });
  Kokkos::Profiling::popRegion();

}

template<typename aobjdType, typename apsdType, typename valT>
void doOneBodyJastrowMultiRatioGrad(aobjdType& aobjd, apsdType& apsd, Kokkos::View<int*>& isValidMap,
			       int numValid, int iel, Kokkos::View<valT**> gradNowView,
				    Kokkos::View<valT*> ratiosView, const Kokkos::HostSpace&) {
  Kokkos::Profiling::pushRegion("obj-evalRatioGrad");
  const int numWalkers = numValid;
  using BarePolicy = Kokkos::TeamPolicy<>;
  BarePolicy pol(numWalkers, Kokkos::AUTO, 32);
  Kokkos::parallel_for("obj-evalRatioGrad-walker-loop", pol,
		       KOKKOS_LAMBDA(BarePolicy::member_type member) {
			 int walkerIdx = member.league_rank();
			 int walkerNum = isValidMap(walkerIdx);
			 auto gv = Kokkos::subview(gradNowView,walkerIdx,Kokkos::ALL());
			 ratiosView(walkerIdx) = aobjd(walkerNum).ratioGrad(member, apsd(walkerNum), iel, gv);
		       });
  Kokkos::Profiling::popRegion();
}

#ifdef KOKKOS_ENABLE_CUDA
template<typename aobjdType, typename apsdType, typename valT>
void doOneBodyJastrowMultiRatioGrad(aobjdType& aobjd, apsdType& apsd, Kokkos::View<int*>& isValidMap,
			       int numValid, int iel, Kokkos::View<valT**> gradNowView,
				    Kokkos::View<valT*> ratiosView, const Kokkos::CudaSpace&) {
  doOneBodyJastrowMultiRatioGrad(aobjd, apsd, isValidMap, numValid, iel, gradNowView, ratiosView);
}

template<typename aobjdType, typename apsdType, typename valT>
void doOneBodyJastrowMultiRatioGrad(aobjdType& aobjd, apsdType& apsd, Kokkos::View<int*>& isValidMap,
			       int numValid, int iel, Kokkos::View<valT**> gradNowView,
				    Kokkos::View<valT*> ratiosView, const Kokkos::CudaUVMSpace&) {
  doOneBodyJastrowMultiRatioGrad(aobjd, apsd, isValidMap, numValid, iel, gradNowView, ratiosView);
}
#endif


/*
template<typename aobjdType, typename apsdType, typename valT>
void doOneBodyJastrowMultiRatioGrad(aobjdType aobjd, apsdType apsd, int iat, 
				    Kokkos::View<valT**> gradNowView,
				    Kokkos::View<valT*> ratiosView) {
  int numWalkers = aobjd.extent(0);
  using BarePolicy = Kokkos::TeamPolicy<>;
  BarePolicy pol(numWalkers, 1, 32);
  //std::cout <<"     in doOneBodyJastrowMultiRatioGrad with iat = " << iat << std::endl;
  Kokkos::parallel_for("obj-evalRatioGrad-walker-loop", pol,
		       KOKKOS_LAMBDA(BarePolicy::member_type member) {
			 int walkerNum = member.league_rank();
			 auto gv = Kokkos::subview(gradNowView,walkerNum,Kokkos::ALL());
			 ratiosView(walkerNum) = aobjd(walkerNum).ratioGrad(member, apsd(walkerNum), iat, gv);
		       });
}
*/
			 
template<typename aobjdType, typename valT>
void doOneBodyJastrowMultiEvalGrad(aobjdType aobjd, int iat, Kokkos::View<valT**> gradNowView) {
  int numWalkers = aobjd.extent(0);
  using BarePolicy = Kokkos::TeamPolicy<>;
  BarePolicy pol(numWalkers, 1, 1);
  Kokkos::parallel_for("obj-evalGrad-walker-loop", pol,
		       KOKKOS_LAMBDA(BarePolicy::member_type member) {
			 int walkerNum = member.league_rank();
			 for (int idim = 0; idim < gradNowView.extent(1); idim++) {
			   gradNowView(walkerNum,idim) = aobjd(walkerNum).Grad(iat,idim);
			 }
		       });
 }

template<typename eiListType, typename apskType, typename aobjdType, typename tempRType,
         typename devRatioType, typename activeMapType>
void doOneBodyJastrowMultiEvalRatio(int pairNum, eiListType& eiList, apskType& apsk,
				    aobjdType& allOneBodyJastrowData,
				    tempRType& unlikeTempR, devRatioType& devRatios, 
				    activeMapType& activeMap, int numActive) {
  int numWalkers = numActive;
  int numKnots = unlikeTempR.extent(1);
  const int numIons = allOneBodyJastrowData(0).Nions(0); // note this is bad, relies on UVM, kill it
  
  Kokkos::parallel_for("obj-multi-ratio", Kokkos::RangePolicy<>(0,numWalkers*numKnots*numIons),
		       KOKKOS_LAMBDA(const int& idx) {
			 const int workingIonNum = idx / numWalkers / numKnots;
			 const int knotNum = (idx - workingIonNum * numWalkers * numKnots) / numWalkers;
			 const int walkerIdx = (idx - workingIonNum * numWalkers * numKnots - knotNum * numWalkers);
			 
			 const int walkerNum = activeMap(walkerIdx);
			 auto& psk = apsk(walkerNum);
			 int iel = eiList(walkerNum, pairNum, 0);
			 auto singleDists = Kokkos::subview(unlikeTempR, walkerNum, knotNum, Kokkos::ALL);
			 allOneBodyJastrowData(walkerIdx).computeU(psk, iel, singleDists, workingIonNum, devRatios, walkerIdx, knotNum);
		       });
  Kokkos::parallel_for("obj-multi-ratio-cleanup", Kokkos::RangePolicy<>(0,numWalkers*numKnots),
		       KOKKOS_LAMBDA(const int& idx) {
			 const int walkerIdx = idx / numKnots;
			 const int knotNum = idx % numKnots;
			 const int walkerNum = activeMap(walkerIdx);
			 if (knotNum == 0) {
			   allOneBodyJastrowData(walkerIdx).updateMode(0) = 0;
			 }
			 int iel = eiList(walkerNum, pairNum, 0);
			 auto val = devRatios(walkerIdx, knotNum);
			 devRatios(walkerIdx,knotNum) = std::exp(allOneBodyJastrowData(walkerIdx).Vat(iel) - val);
		       });



  /*
  using BarePolicy = Kokkos::TeamPolicy<>;
  BarePolicy pol(numWalkers, Kokkos::AUTO, 32);
  
  Kokkos::parallel_for("obj-multi-ratio", pol,
		       KOKKOS_LAMBDA(BarePolicy::member_type member) {
			 int walkerIndex = member.league_rank();
			 int walkerNum = activeWalkerIdx(walkerIndex);
			 auto& psk = apsk(walkerNum);
			 //ParticleSetKokkos& psk = apsk(walkerNum);
			 //auto& jd = allOneBodyJastrowData(walkerIndex);
			 
			 allOneBodyJastrowData(walkerIndex).updateMode(0) = 0;

			 Kokkos::parallel_for(Kokkos::TeamThreadRange(member, numKnots),
					      [=](const int& knotNum) {
						auto singleDists = Kokkos::subview(unlikeTempR, walkerNum, knotNum, Kokkos::ALL);
						auto val = allOneBodyJastrowData(walkerIndex).computeU(member, psk, singleDists);
						int iat = eiList(walkerNum, pairNum, 1);
						devRatios(walkerIndex, numKnots) = std::exp(allOneBodyJastrowData(walkerIndex).Vat(iat) - val);
					      });
		       });
  */
}
 
template<typename aobjdType, typename apsdType, typename valT>
void doOneBodyJastrowMultiEvaluateLog(aobjdType aobjd, apsdType apsd, Kokkos::View<valT*> values) {
  Kokkos::Profiling::pushRegion("1BJ-multiEvalLog");
  const int numWalkers = aobjd.extent(0);
  using BarePolicy = Kokkos::TeamPolicy<>;
  const int numElectrons = aobjd(0).Nelec(0);
  BarePolicy pol(numWalkers*numElectrons, 1, 32);
  Kokkos::parallel_for("obj-evalLog-waker-loop", pol,
		       KOKKOS_LAMBDA(BarePolicy::member_type member) {
			 int walkerNum = member.league_rank()/numElectrons; 
			 values(walkerNum) = aobjd(walkerNum).evaluateLog(member, apsd(walkerNum));
		       });
  Kokkos::Profiling::popRegion();
}
  
/** @ingroup WaveFunctionComponent
 *  @brief Specialization for one-body Jastrow function using multiple functors
 */
template<class FT>
struct OneBodyJastrow : public WaveFunctionComponent
{
  using jasDataType = OneBodyJastrowKokkos<typename FT::real_type, OHMMS_DIM>;
  jasDataType jasData;
  bool splCoefsNotAllocated;

  /// alias FuncType
  using FuncType = FT;
  /// type of each component U, dU, d2U;
  using valT = typename FT::real_type;
  /// element position type
  using posT = TinyVector<valT, OHMMS_DIM>;
  /// use the same container
  using RowContainer = DistanceTableData::RowContainer;
  /// table index
  int myTableID;
  /// number of ions
  int Nions;
  /// number of electrons
  int Nelec;
  /// number of groups
  int NumGroups;
  /// reference to the sources (ions)
  const ParticleSet& Ions;

  ///\f$Vat[i] = sum_(j) u_{i,j}\f$
  Vector<valT> Vat;
  Kokkos::View<valT*> U, dU, d2U;
  Kokkos::View<valT*> DistCompressed;
  Kokkos::View<int*> DistIndice;

  OneBodyJastrow(const ParticleSet& ions, ParticleSet& els) : Ions(ions)
  {
    initalize(els);
    myTableID                 = els.addTable(ions, DT_SOA);
    WaveFunctionComponentName = "OneBodyJastrow";
  }

  OneBodyJastrow(const OneBodyJastrow& rhs) = default;

  ~OneBodyJastrow()
  {
 //   for (int i = 0; i < F.size(); ++i)
 //     if (F[i] != nullptr)
 //       delete F[i];
  }

  /* initialize storage */
  void initalize(ParticleSet& els)
  {
    Nions     = Ions.getTotalNum();
    NumGroups = Ions.getSpeciesSet().getTotalNum();

    if (NumGroups > 1 && !Ions.IsGrouped)
    {
      NumGroups = 0;
    }
    Nelec = els.getTotalNum();

    U              = Kokkos::View<valT*>("U",Nions);
    dU             = Kokkos::View<valT*>("dU",Nions);
    d2U            = Kokkos::View<valT*>("d2U",Nions);
    DistCompressed = Kokkos::View<valT*>("DistCompressed",Nions);
    DistIndice     = Kokkos::View<int*>("DistIndice",Nions);

    initializeJastrowKokkos();
  }

  void initializeJastrowKokkos() {
    jasData.LogValue       = Kokkos::View<valT[1]>("LogValue");

    jasData.Nelec          = Kokkos::View<int[1]>("Nelec");
    auto NelecMirror       = Kokkos::create_mirror_view(jasData.Nelec);
    NelecMirror(0)         = Nelec;
    Kokkos::deep_copy(jasData.Nelec, NelecMirror);
    
    jasData.Nions          = Kokkos::View<int[1]>("Nions");
    auto NionsMirror       = Kokkos::create_mirror_view(jasData.Nions);
    NionsMirror(0)         = Nions;
    Kokkos::deep_copy(jasData.Nions, NionsMirror);
    
    jasData.NumGroups      = Kokkos::View<int[1]>("NumGroups");
    auto NumGroupsMirror   = Kokkos::create_mirror_view(jasData.NumGroups);
    NumGroupsMirror(0)     = NumGroups;
    Kokkos::deep_copy(jasData.NumGroups, NumGroupsMirror);
    
    jasData.updateMode     = Kokkos::View<int[1]>("updateMode");
    auto updateModeMirror  = Kokkos::create_mirror_view(jasData.updateMode);
    updateModeMirror(0)    = 3;
    Kokkos::deep_copy(jasData.updateMode, updateModeMirror);
    
    jasData.temporaryScratch = Kokkos::View<valT[1]>("temporaryScratch");
    jasData.temporaryScratchDim = Kokkos::View<valT[OHMMS_DIM]>("temporaryScratchDim");

    // these things are just zero on the CPU, so don't have to set their values
    jasData.curGrad        = Kokkos::View<valT[OHMMS_DIM]>("curGrad");
    jasData.Grad           = Kokkos::View<valT*[OHMMS_DIM]>("Grad", Nelec);
    jasData.curLap         = Kokkos::View<valT[1]>("curLap");
    jasData.Lap            = Kokkos::View<valT*>("Lap", Nelec);
    jasData.curAt          = Kokkos::View<valT[1]>("curAt");
    jasData.Vat            = Kokkos::View<valT*>("Vat", Nelec);
    
    // these things are already views, so just do operator=
    jasData.U              = U;
    jasData.dU             = dU;
    jasData.d2U            = d2U;
    jasData.DistCompressed = DistCompressed;
    jasData.DistIndices    = DistIndice;
    
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
    auto dAmirror           = Kokkos::create_mirror_view(jasData.dA);
    jasData.d2A            = Kokkos::View<valT[16]>("d2A");
    auto d2Amirror           = Kokkos::create_mirror_view(jasData.d2A);

    for (int i = 0; i < 16; i++) {
      Amirror(i) = A[i];
      dAmirror(i) = dA[i];
      d2Amirror(i) = d2A[i];
    }
    Kokkos::deep_copy(jasData.A, Amirror);
    Kokkos::deep_copy(jasData.dA, dAmirror);
    Kokkos::deep_copy(jasData.d2A, d2Amirror);


    // also set up and allocate memory for cutoff_radius, DeltaRInv
    jasData.cutoff_radius   = Kokkos::View<valT*>("Cutoff_Radii", NumGroups);
    jasData.DeltaRInv       = Kokkos::View<valT*>("DeltaRInv", NumGroups);

    // unfortunately have to defer setting up SplineCoefs because we don't yet know
    // how many elements are in SplineCoefs on the cpu
    splCoefsNotAllocated   = true;
  }

  void addFunc(int source_type, FT* afunc, int target_type = -1)
  {
    // also copy in cutoff_radius, DeltaRInv
    auto crMirror = Kokkos::create_mirror_view(jasData.cutoff_radius);
    auto drinvMirror = Kokkos::create_mirror_view(jasData.cutoff_radius);
    Kokkos::deep_copy(crMirror, jasData.cutoff_radius);
    Kokkos::deep_copy(drinvMirror, jasData.DeltaRInv);
    crMirror(source_type) = afunc->cutoff_radius;
    drinvMirror(source_type) = afunc->DeltaRInv;
    Kokkos::deep_copy(jasData.cutoff_radius, crMirror);
    Kokkos::deep_copy(jasData.DeltaRInv, drinvMirror);

    //if necessary set up SplineCoefs view on device and then copy data it
    if (splCoefsNotAllocated) {
      splCoefsNotAllocated = false;
      jasData.SplineCoefs   = Kokkos::View<valT**>("SplineCoefficients", NumGroups, afunc->SplineCoefs.extent(0));
    }
    auto bigScMirror   = Kokkos::create_mirror_view(jasData.SplineCoefs);
    auto smallScMirror = Kokkos::create_mirror_view(afunc->SplineCoefs);
    Kokkos::deep_copy(smallScMirror, afunc->SplineCoefs);
    Kokkos::deep_copy(bigScMirror,   jasData.SplineCoefs);
    for (int i = 0; i < afunc->SplineCoefs.extent(0); i++) {
      bigScMirror(source_type, i) = smallScMirror(i);
    }
    Kokkos::deep_copy(jasData.SplineCoefs, bigScMirror);
  }



  /////////// Helpers to populate collective data structures
  template<typename aobjType, typename apsdType, typename vectorType, typename vectorType2>
  void populateCollectiveViews(aobjType aobjd, apsdType apsd, vectorType& WFC_list, vectorType2& P_list) {
    auto aobjdMirror = Kokkos::create_mirror_view(aobjd);
    auto apsdMirror = Kokkos::create_mirror_view(apsd);

    for (int i = 0; i < WFC_list.size(); i++) {
      aobjdMirror(i) = static_cast<OneBodyJastrow*>(WFC_list[i])->jasData;
      apsdMirror(i) = P_list[i]->psk;
    }
    Kokkos::deep_copy(aobjd, aobjdMirror);
    Kokkos::deep_copy(apsd, apsdMirror);
  }

  template<typename aobjType, typename apsdType, typename vectorType, typename vectorType2>
  void populateCollectiveViews(aobjType aobjd, apsdType apsd, vectorType& WFC_list, vectorType2& P_list, const std::vector<bool>& isAccepted) {
    auto aobjdMirror = Kokkos::create_mirror_view(aobjd);
    auto apsdMirror = Kokkos::create_mirror_view(apsd);

    int idx = 0;
    for (int i = 0; i < WFC_list.size(); i++) {
      if (isAccepted[i]) {
	aobjdMirror(idx) = static_cast<OneBodyJastrow*>(WFC_list[i])->jasData;
	apsdMirror(idx) = P_list[i]->psk;
	idx++;
      }
    }
    Kokkos::deep_copy(aobjd, aobjdMirror);
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
                                 ParticleSet::ParticleValue_t& values) {
    if (WFC_list.size() > 0) {
      // make a view of all of the OneBodyJastrowData and relevantParticleSetData
      Kokkos::View<jasDataType*> allOneBodyJastrowData("aobjd", WFC_list.size()); 
      Kokkos::View<ParticleSet::pskType*> allParticleSetData("apsd", P_list.size());
      populateCollectiveViews(allOneBodyJastrowData, allParticleSetData, WFC_list, P_list);
      
      // need to make a view to hold all of the output LogValues
      Kokkos::View<valT*> tempValues("tempValues", P_list.size());
      
      // need to write this function
      doOneBodyJastrowMultiEvaluateLog(allOneBodyJastrowData, allParticleSetData, tempValues);
      
      // copy the results out to values
      auto tempValMirror = Kokkos::create_mirror_view(tempValues);
      Kokkos::deep_copy(tempValMirror, tempValues);
      
      for (int i = 0; i < P_list.size(); i++) {
	values[i] = tempValMirror(i);
      }
    }
  }
  
  virtual void multi_evaluateLog(const std::vector<WaveFunctionComponent*>& WFC_list,
				 WaveFunctionKokkos& wfc,
				 Kokkos::View<ParticleSet::pskType*>& psk,
				 ParticleSet::ParticleValue_t& values) {
    // need to write this function
    doOneBodyJastrowMultiEvaluateLog(wfc.oneBodyJastrows, psk, wfc.ratios_view);
      
    Kokkos::deep_copy(wfc.ratios_view_mirror, wfc.ratios_view);
    
    for (int i = 0; i < WFC_list.size(); i++) {
      values[i] = wfc.ratios_view_mirror(i);
    }
  }
   

  virtual void multi_evalGrad(const std::vector<WaveFunctionComponent*>& WFC_list,
			      const std::vector<ParticleSet*>& P_list,
                              int iat,
			      std::vector<posT>& grad_now) {

    // make a view of all of the OneBodyJastrowData and relevantParticleSetData
    Kokkos::View<jasDataType*> allOneBodyJastrowData("aobjd", WFC_list.size()); 
    Kokkos::View<ParticleSet::pskType*> allParticleSetData("apsd", P_list.size());
    populateCollectiveViews(allOneBodyJastrowData, allParticleSetData, WFC_list, P_list);
    
    // need to make a view to hold all of the output LogValues
    Kokkos::View<RealType**> grad_now_view("tempValues", P_list.size(), OHMMS_DIM);

    // need to write this function
    doOneBodyJastrowMultiEvalGrad(allOneBodyJastrowData, iat, grad_now_view);

    // copy the results out to values
    auto grad_now_view_mirror = Kokkos::create_mirror_view(grad_now_view);
    Kokkos::deep_copy(grad_now_view_mirror, grad_now_view);
    
    for (int i = 0; i < P_list.size(); i++) {
      for (int j = 0; j < OHMMS_DIM; j++) {
	grad_now[i][j] = grad_now_view_mirror(i,j);
      }
    }
  }

  virtual void multi_evalGrad(const std::vector<WaveFunctionComponent*>& WFC_list,
                              WaveFunctionKokkos& wfc,
                              Kokkos::View<ParticleSet::pskType*> psk,
                              int iat,
                              std::vector<PosType>& grad_now) {
    const int numItems = WFC_list.size();
    doOneBodyJastrowMultiEvalGrad(wfc.oneBodyJastrows, iat, wfc.grad_view);

    Kokkos::deep_copy(wfc.grad_view_mirror, wfc.grad_view);
    
    for (int i = 0; i < numItems; i++) {
      for (int j = 0; j < OHMMS_DIM; j++) {
	grad_now[i][j] = wfc.grad_view_mirror(i,j);
      }
    }
  }

  virtual void multi_ratioGrad(const std::vector<WaveFunctionComponent*>& WFC_list,
                               WaveFunctionKokkos& wfc,
                               Kokkos::View<ParticleSet::pskType*> psk,
                               int iel,
			       Kokkos::View<int*>& isValidMap, int numValid,
			       std::vector<ValueType>& ratios,
                               std::vector<PosType>& grad_new) {
    Kokkos::Profiling::pushRegion("obj-multi_ratioGrad");
    if (numValid > 0) {
      Kokkos::Profiling::pushRegion("obj-multi_ratioGrad::kernel");
      doOneBodyJastrowMultiRatioGrad(wfc.oneBodyJastrows, psk, isValidMap, numValid, iel, 
				     wfc.grad_view, wfc.ratios_view, typename Kokkos::View<int*>::memory_space());
      Kokkos::fence();
      Kokkos::Profiling::popRegion();

      Kokkos::Profiling::pushRegion("obj-multi_ratioGrad::copyOut");
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
      Kokkos::Profiling::popRegion();
    }
    Kokkos::Profiling::popRegion();
    //std::cout << "      finishing J1 multi_ratioGrad" << std::endl;

  }
      


  /*
  virtual void multi_ratioGrad(const std::vector<WaveFunctionComponent*>& WFC_list,
			       const std::vector<ParticleSet*>& P_list,
			       int iat,
			       std::vector<valT>& ratios,
			       std::vector<posT>& grad_new) {
    //std::cout << "      starting J1 multi_ratioGrad" << std::endl;
    if (WFC_list.size() > 0) {

      // make a view of all of the OneBodyJastrowData and relevantParticleSetData
      Kokkos::View<jasDataType*> allOneBodyJastrowData("aobjd", WFC_list.size()); 
      Kokkos::View<ParticleSet::pskType*> allParticleSetData("apsd", P_list.size());
      populateCollectiveViews(allOneBodyJastrowData, allParticleSetData, WFC_list, P_list);
      //std::cout << "       finished populating collective views" << std::endl;

      // need to make a view to hold all of the output LogValues
      Kokkos::View<RealType**> grad_new_view("tempValues", P_list.size(), OHMMS_DIM);
      Kokkos::View<RealType*> ratios_view("ratios", P_list.size());
      
      // need to write this function
      //std::cout << "       starting doOneBJMultiRatioGrad" << std::endl;
      doOneBodyJastrowMultiRatioGrad(allOneBodyJastrowData, allParticleSetData, iat, grad_new_view, ratios_view);
      Kokkos::fence();
      //std::cout << "       finishing doOneBJMultiRatioGrad" << std::endl;      

      // copy the results out to values
      auto grad_new_view_mirror = Kokkos::create_mirror_view(grad_new_view);
      Kokkos::deep_copy(grad_new_view_mirror, grad_new_view);
      auto ratios_view_mirror = Kokkos::create_mirror_view(ratios_view);
      Kokkos::deep_copy(ratios_view_mirror, ratios_view);
      //std::cout << "       finished copying grad and ratios out" << std::endl;

      for (int i = 0; i < P_list.size(); i++) {
	ratios[i] = ratios_view_mirror(i);
	for (int j = 0; j < OHMMS_DIM; j++) {
	  grad_new[i][j] += grad_new_view_mirror(i,j);
	}
      }
    }
    //std::cout << "      finishing J1 multi_ratioGrad" << std::endl;

  }
  */

  virtual void multi_acceptrestoreMove(const std::vector<WaveFunctionComponent*>& WFC_list,
				       WaveFunctionKokkos& wfc,
				       Kokkos::View<ParticleSet::pskType*> psk,
				       Kokkos::View<int*>& isAcceptedMap, int numAccepted, int iel) {
    doOneBodyJastrowMultiAcceptRestoreMove(wfc.oneBodyJastrows, psk, isAcceptedMap, numAccepted, iel);
  }


  virtual void multi_acceptRestoreMove(const std::vector<WaveFunctionComponent*>& WFC_list,
				       const std::vector<ParticleSet*>& P_list,
				       const std::vector<bool>& isAccepted,
				       int iat) {
    int numAccepted = 0;
    for (int i = 0; i < isAccepted.size(); i++) {
      if (isAccepted[i]) {
	numAccepted++;
      }
    }
    if (numAccepted > 0) {

      // make a view of all of the OneBodyJastrowData and relevantParticleSetData
      Kokkos::View<jasDataType*> allOneBodyJastrowData("aobjd", numAccepted); 
      Kokkos::View<ParticleSet::pskType*> allParticleSetData("apsd", numAccepted);
      populateCollectiveViews(allOneBodyJastrowData, allParticleSetData, WFC_list, P_list, isAccepted);
      
      // need to write this function
      doOneBodyJastrowMultiAcceptRestoreMove(allOneBodyJastrowData, allParticleSetData, iat);
    }
    // be careful on this one, looks like it is being done for side effects.  Should see what needs to go back!!!
  }

  virtual void multi_evalRatio(int pairNum, Kokkos::View<int***>& eiList,
			       WaveFunctionKokkos& wfc,
			       Kokkos::View<ParticleSetKokkos<RealType, ValueType, 3>*>& apsk,
			       Kokkos::View<RealType***>& likeTempR,
			       Kokkos::View<RealType***>& unlikeTempR,
			       std::vector<ValueType>& ratios, int numActive) {
    const int numKnots = likeTempR.extent(1);
    
    doOneBodyJastrowMultiEvalRatio(pairNum, eiList, apsk, wfc.oneBodyJastrows, unlikeTempR, 
				   wfc.knots_ratios_view, wfc.activeMap, numActive);

    Kokkos::deep_copy(wfc.knots_ratios_view_mirror, wfc.knots_ratios_view);
    for (int i = 0; i < numActive; i++) {
      const int walkerNum = wfc.activeMapMirror(i);
      for (int j = 0; j < wfc.knots_ratios_view_mirror.extent(1); j++) {
	ratios[walkerNum*numKnots+j] = wfc.knots_ratios_view_mirror(i,j);
      }
    }
  }

    
    

  virtual void multi_evaluateGL(const std::vector<WaveFunctionComponent*>& WFC_list,
				const std::vector<ParticleSet*>& P_list,
				const std::vector<ParticleSet::ParticleGradient_t*>& G_list,
				const std::vector<ParticleSet::ParticleLaplacian_t*>& L_list,
				bool fromscratch = false) {
    
    // make a view of all of the OneBodyJastrowData and relevantParticleSetData
    Kokkos::View<jasDataType*> allOneBodyJastrowData("aobjd", WFC_list.size()); 
    Kokkos::View<ParticleSet::pskType*> allParticleSetData("apsd", P_list.size());
    populateCollectiveViews(allOneBodyJastrowData, allParticleSetData, WFC_list, P_list);

    doOneBodyJastrowMultiEvaluateGL(allOneBodyJastrowData, allParticleSetData, Nelec, fromscratch);

    // know that we will need LogValue to up updated after this, possibly other things in ParticleSet!!!
    for (int i = 0; i < WFC_list.size(); i++) {
      auto LogValueMirror = Kokkos::create_mirror_view(static_cast<OneBodyJastrow*>(WFC_list[i])->jasData.LogValue);
      Kokkos::deep_copy(LogValueMirror, static_cast<OneBodyJastrow*>(WFC_list[i])->jasData.LogValue);
      LogValue = LogValueMirror(0);
    }
  }
  
  
  ///////////////////////// end internal multi functions

};

} // namespace qmcplusplus
#endif
