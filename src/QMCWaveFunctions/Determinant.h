////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2017 QMCPACK developers.
//
// File developed by: M. Graham Lopez
//
// File created by: M. Graham Lopez
////////////////////////////////////////////////////////////////////////////////
// -*- C++ -*-

/**
 * @file Determinant.h
 * @brief Determinant piece of the wave function
 */

// need to implement: multi_evaluateLog, multi_evaluateGL, multi_evalGrad,
//                    multi_ratioGrad, multi_acceptrestoreMove (under the hood calls acceptMove)

// strategy is to make a View of DiracDeterminantKokkos and then to write functions
// that contain kernels that work on them.  This will allow us to follow the pointers
// during the parallel evaluation and potentially do more parallel work

#ifndef QMCPLUSPLUS_DETERMINANT_H
#define QMCPLUSPLUS_DETERMINANT_H

#include <Kokkos_Core.hpp>
#include <impl/Kokkos_Timer.hpp>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#ifdef KOKKOS_ENABLE_CUDA
#include "cublas_v2.h"
#include "cusolverDn.h"
#endif

#include "QMCWaveFunctions/WaveFunctionComponent.h"
#include "Numerics/LinAlgKokkos.h"
//#include "Utilities/RandomGenerator.h"


namespace qmcplusplus
{


template<class linAlgHelperType, typename ValueType>
ValueType InvertWithLog(DiracDeterminantKokkos& ddk, LinAlgHelperType& lah, ValueType& phase) {
  ValueType locLogDet(0.0);
  lah.getrf(ddk.psiM);
  auto locPiv = lah.getPivot(); // note, this is in device memory
  int sign_det = 1;

  Kokkos::parallel_reduce(view.extent(0), KOKKOS_LAMBDA ( int ii, int& cur_sign) {
      cur_sign = (locPiv(i) == i+1) ? 1 : -1;
      cur_sign *= (ddk.psiM(i,i) > 0) ? 1 : -1;
    }, Kokkos::Prod<int>(sign_det));

  Kokkos::parallel_reduce(view.extent(0), KOKKOS_LAMBDA (int ii, ValueType& v) {
      v += std::log(std::abs(ddk.psiM(i,i)));
    }, logdet);
  lah.getri(ddk.psiM);
  phase = (sign_det > 0) ? 0.0 : M_PI;
  //auto logValMirror = Kokkos::create_mirror_view(ddk.LogValue);
  //logValMirror(0) = locLogDet;
  //Kokkos::deep_copy(ddk.LogValue, logValMirror);
  return locLogDet;
}

template<class linAlgHelperType, typename ValueType>
void updateRow(DiracDeterminantKokkos& ddk, LinAlgHelperType& lah, int rowchanged, ValueType c_ratio_in) {
  constexpr ValueType cone(1.0);
  constexpr ValueType czero(0.0);
  ValueType c_ratio = cone / c_ratio_in;
  Kokkos::Profiling::PushRegion("updateRow::gemvTrans");
  lah.gemvTrans(ddk.psiMinv, ddk.psiV, ddk.tempRowVec, c_ratio, czero);
  Kokkos::Profiling::popRegion();

  // hard work to modify one element of temp on the device
  Kokkos::Profiling::pushRegion("updateRow::pokeSingleValue");
  auto devElem = subview(ddk.tempRowVec, rowchanged);
  auto devElem_mirror = Kokkos::create_mirror_view(devElem);
  devElem_mirror(0) = cone - c_ratio;
  Kokkos::deep_copy(devElem, devElem_mirror);
  Kokkos::Profiling::popRegion();

  Kokkos::Profiling::pushRegion("updateRow::populateRcopy");
  lah.copyChangedRow(rowchanged, ddk.psiMinv, ddk.rcopy);
  Kokkos::Profiling::popRegion();

  // now do ger
  Kokkos::Profiling::pushRegion("updateRow::ger");
  lah.ger(ddk.psiMinv, ddk.rcopy, ddk.tempRowVec, -cone);
  Kokkos::Profiling::popRegion();
}

template<typename memorySpace, typename addkType, typename vectorType, typename resVecType>
void doDiracDeterminantMultiEvaluateLog(ddkType addk, vectorType& wfcv, resVecType& results) {
  // for each element in addk,
  //      1. copy transpose of psiMsave to psiM
  //      2. invert psiM
  //      3. copy psiM to psiMinv

  // 1. copy transpose of psiMsave to psiM for all walkers
  const int numWalkers = addk.extent(0);
  const int numEls = static_cast<DiracDeterminant*>(wfcv[i])->ddk.psiV.extent(0);
  Kokkos::parallel_for("elementWiseCopyTransAllPsiM",
		       Kokkos::MDRangePolicy<Kokkos::Rank<3,Kokkos::Iterate::Left> >({0,0,0}, {numWalkers, numEls, numEls}),
		       KOKKOS_LAMBDA(const int& i0, const int& i1, const int& i2) {
			 addk(i0).psiM(i1, i2) = addk(i0).psiMsave(i2,i1);
		       });
  Kokkos::fence();

  // 2. invert psiM.  This will loop over the walkers and invert each psiM.  Need to switch to a batched version of this
  // simplest thing would be to assume mkl and then have this be a kokkos parallel_for, inside of which you would call
  // mkl_set_num_threads_local before doing the linear algebra calls (wouldn't be able to use lah because couldn't follow the
  // pointer inside the kernel
  for (int i = 0; i < addk.extent(0); i++) {
    auto toInv = static_cast<DiracDeterminant*>(wfcv[i])->ddk.psiM;
    static_cast<DiracDeterminant*>(wfcv[i])->lah.invertMatrix(toInv);
  }

  // 3. copy psiM to psiMinv
  Kokkos::parallel_for("elementWiseCopyAllPsiM",
		       Kokkos::MDRangePolicy<Kokkos::Rank<3,Kokkos::Iterate::Left> >({0,0,0}, {numWalkers, numEls, numEls}),
		       KOKKOS_LAMBDA(const int& i0, const int& i1, const int& i2) {
			 addk(i0).psiMinv(i1, i2) = addk(i0).psiM(i1,i2);
		       });
  Kokkos::fence();

  for (int i = 0; i < results.size(); i++) {
    results[i] = 0.0;
  }
}


#ifdef KOKKOS_ENABLE_CUDA
// this is specialized to when the Views live on the GPU in CudaSpace 
template<typename addkType, typename vectorType, typename resVecType>
void doDiracDeterminantMultiEvaluateLog<Kokkos::CudaSpace>(ddkType addk, vectorType& wfcv, resVecType& results) {
  // for each element in addk,
  //      1. copy transpose of psiMsave to psiM
  //      2. invert psiM
  //      3. copy psiM to psiMinv

  using ValueType = static_cast<DiracDeterminant*>(wfcv[0])->ddk.psiM::data_type;
  const int numWalkers = addk.extent(0);
  const int numEls = static_cast<DiracDeterminant*>(wfcv[0])->ddk.psiV.extent(0);
  
  // 1. copy transpose of psiMsave to psiM for all walkers and also zero out temp matrices
  Kokkos::parallel_for("elementWiseCopyTransAllPsiM",
		       Kokkos::MDRangePolicy<Kokkos::Rank<3,Kokkos::Iterate::Left> >({0,0,0}, {numWalkers, numEls, numEls}),
		       KOKKOS_LAMBDA(const int& i0, const int& i1, const int& i2) {
			 addk(i0).psiM(i1, i2) = addk(i0).psiMsave(i2,i1);
			 addk(i0).getRiWorkSpace(i1,i2) = 0.0;
		       });
  Kokkos::fence();


   // 2. invert psiM.  This will loop over the walkers and invert each psiM.  Need to switch to a batched version of this
  // simplest thing would be to assume mkl and then have this be a kokkos parallel_for, inside of which you would call
  // mkl_set_num_threads_local before doing the linear algebra calls

  // set up temp spaces ahead of calls
  Kokkos::parallel_for("makeIntoIdentity",
		       Kokkos::MDRangePolicy<Kokkos::Rank<2,Kokkos::Iterate::Left> >({0,0},{numWalkers,numEls}),
		       KOKKOS_LAMBDA(const int& i0, const int& i1) {
			 addk(i0).getRiWorkspace(i1,i1) = 1.0;
		       });
  Kokkos::fence();
    

  cudaStream_t *streams = (cudaStream_t *) malloc(addk.extent(0)*sizeof(cudaStream_t));
  for (int i = 0; i < numWalkers; i++) {
    cudaStreamCreate(&streams[i]);
  }
  
  // unfortunately, need to access these through the vector anyway, cannot do addk(i).psiM on host!
  for (int i = 0; i < numWalkers; i++) {
    auto& lahref = static_cast<DiracDeterminant*>(wfcv[i])->lah;
    cusolverDNSetStream(lahref.cusolver_handle, streams[i]);
    auto& psiM = static_cast<DiracDeterminant*>(wfcv[i])->ddk.psiM;
 
    auto& getRfWs = static_cast<DiracDeterminant*>(wfcv[i])->ddk.getRfWorkSpace;
    auto& getRiWs = static_cast<DiracDeterminant*>(wfcv[i])->ddk.getRiWorkSpace;
    auto& piv = static_cast<DiracDeterminant*>(wfcv[i])->ddk.piv;

    int info;
    getrf_gpu_impl(psiM.extent(0), psiM.extent(1),
		   lahref.pointerConverter(psiM.data()), psiM.extent(0),
		   lahref.pointerConverter(getRfWs.data()),
		   piv.data(), info, lahref.cusolver_handle);
    getri_gpu_impl(psiM.extent(0), lahref.pointerConverter(psiM.data()), piv.data,
		   lahref.pointerConverter(getRiWs.data()), info, lahref.cusolver_handle);
    // Results are sitting in getRiWs now!!!
    
  }
  cudaDeviceSynchronize();
  Kokkos::fence();  

  for (int i =0; i < numWalkers; i++) {
    cudaStreamDestroy(streams[i]);
  }

  // 3. copy getRiWs to psiM and to psiMinv
  Kokkos::parallel_for("elementWiseCopyAllPsiM",
		       Kokkos::MDRangePolicy<Kokkos::Rank<3,Kokkos::Iterate::Left> >({0,0,0}, {addk.extent(0), addk(0).psiM.extent(0), addk(0).PsiM.extent(1)}),
		       KOKKOS_LAMBDA(const int& i0, const int& i1, const int& i2) {
			 addk(i0).psiM(i1,i2) = addk(i0).getRiWorkSpace(i1,i2);
			 addk(i0).psiMinv(i1, i2) = addk(i0).getRiWorkSpace(i1,i2);
		       });
  Kokkos::fence();

  for (int i = 0; i < results.size(); i++) {
    results[i] = 0.0;
  }
}
#endif

template<typename addkType, typename vectorType, typename resVecType>
void doDiracDeterminantMultiEvalRatio(addkType addk, vectorType& wfcv, resVecType& ratios, int iel) {
  using ValueType = resVecType::data_type;
  const int numEls = static_cast<DiracDeterminant*>(wfcv[0])->ddk.psiV.extent(0);
  const int numWalkers = addk.extent(0);
  constexpr double shift(0.5);

  // could do this in parallel, but the random number stream wouldn't be the same.
  // could avoid the data transfer that way.  This is OK, because the real code would
  // be calling evaluateV on the sposet and that would happen on the device as well.
  for (int i = 0; i < numWalkers; i++) {
    auto& psiV = static_cast<DiracDeterminant*>(WFC_list[i])->ddk.psiV;
    auto psiVMirror = Kokkos::make_mirror_view(psiV);
    for (int j = 0; j < numEls; j++) {
      psiVMirror(j) = static_cast<DiracDeterminant*>(WFC_list[0]).myRandom() - shift;
    }
    Kokkos::deep_copy(psiV, psiVMirror);
  }

  using BarePolicy = Kokkos::TeamPolicy<>;
  BarePolicy pol(numWalkers, 1, 32);
  Kokkos::parallel_for("dd-evalRatio-walker-loop", pol,
		       KOKKOS_LAMBDA(BarePolicy::member_type member) {
			 int walkerNum = member.league_rank();
			 ValueType sumOver = 0.0;
			 Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(member, numEls),
						 [=] (const int& i, ValueType& innersum) {
						   const idx = iel - addk(walkerNum).FirstIndex(0);
						   innersum += addk(walkerNum).psiV(i) * addk(walkerNum).psiMinv(idx,i); 
						 }, sumOver);
			 Kokkos::single(Kokkos::PerTeam(member), [=]() {
			     ratios(walkerNum) = sumOver;
			   });
		       });
}

template<typename addkType, typename awType, typename eiListType, typename psiVType, typename ratiosType, typename rngType>
void doDiracDeterminantMultiEvalRatio(int pairNum, addkType& addk, awType& activeWalkers, eiListType& eiList, psiVType& psiVScratch, ratiosType& ratios, rngType& rng) {
  using ValueType = psiVType::data_type;
  const int numEls = psiVScratch.extent(2);
  const int numKnots = psiVScratch.extent(1);
  const int numWalkers = activeWalkers.extent(0);
  constexpr double shift(0.5);

  // currently expecting psiVScratch to have been filled by a previous call to spo.multi_evaluate_v
  /*
  auto psiVScratchMirror = Kokkos::create_mirror_view(psiVScratch);
  for(int iw = 0; iw < numWalkers; iw++) {
    for (int knot = 0; knot < numKnots; knot++) {
      for (int el = 0; el < numEls; el++) {
	psiVScratchMirror(iw,knot,iel) = rng() - shift;
      }
    }
  }
  Kokkos::deep_copy(psiVScratch, psiVScratchMirror);
  */

  Kokkos::View<ValueType**> results("ratios", numWalkers, numKnots);

  // note, there should be some scope for caching as psiMinv does not depend on knotNum
  using BarePolicy = Kokkos::TeamPolicy<>;
  BarePolicy pol(numWalkers, Kokkos::AUTO, 32);
  Kokkos::parallel_for("dd-evalRatio-general", pol,
		       KOKKOS_LAMBDA(BarePolicy::member_type member) {
			 const int walkerNum = activeWalkers(member.league_rank());
			 const int bandIdx = eiList(walkerNum, pairNum, 0) - addk(walkerNum).FirstIndex(0);
			 Kokkos::parallel_for("loopOverKnots", Kokkos::TeamThreadRange(member, numKnots),
					      [=] (const int& knotNum) {
						Kokkos::parallel_reduce("loopOverEls", Kokkos::ThreadVectorRange(member, numEls),
									[=] (const int& i, ValueType& innersum) {
									  innersum += psiVScratchMirror(walkerNum,knotNum,i) *
									    addk(walkerNum).psiMinv(bandIdx,i);
									}, results(walkerNum, knotNum));
					      });
		       });

  auto resultsMirror = Kokkos::create_mirror_view(results);
  Kokkos::deep_copy(resultsMirror, results);
  for (int i = 0; i < resultsMirror.extent(0); i++) {
    for (int j = 0; j < resultsMirror.extent(1); j++) {
      ratios[i*numKnots+j] = resultsMirror(i,j);
    }
  }
}

template<typename memory_space, typename addkType, typename vectorType>
void doDiracDeterminantMultiAccept(addkType& addk, vectorType& WFC_list, int iel) {
  // for every walker, need to do updateRow followed by copyBack
  int numWalkers = WFC_list.size();
  int numEls = static_cast<DiracDeterminant*>(WFC_list[0])->psiV.extent(0);
  using ValueType = static_cast<DiracDeterminant*>(wfcv[0])->ddk.psiM::data_type;
  int rowChanged = iel-static_cast<DiracDeterminant*>(WFC_list[0]).FirstIndex;
  constexpr ValueType cone(1.0);
  constexpr ValueType czero(0.0);

  // 1. gemvTrans
  Kokkos::Profiling::PushRegion("updateRow::gemvTrans");
  for (int i = 0; i < numWalkers; i++) {
    DiracDeterminant* ddp = static_cast<DiracDeterminant*>(WFC_list[i]);
    DiracDeterminantKokkos& ddk = ddp->ddk;
    const ValueType c_ratio = cone / ddp->curRatio; 
    ddp->lah.gemvTrans(ddk.psiMinv, ddk.psiV, ddk.tempRowVec, c_ratio, czero); 
  }
  Kokkos::Profiling::popRegion();
  
  // 2. poke one element on the device for each walker
  Kokkos::Profiling::pushRegion("updateRow::pokeSingleValue");
  Kokkos::View<ValueType*> poke("poke", numWalkers);
  auto pokeView = Kokkos:create_mirror_view(poke);
  for (int i = 0; i < numWalkers; i++) {
    DiracDeterminant* ddp = static_cast<DiracDeterminant>(WFC_list[i]);
    pokeView(i) = cone - cone / ddp->curRatio;
  }
  Kokkos::deep_copy(poke, pokeView);
  Kokkos::parallel_for(numWalkers, Kokkos_LAMBDA(int i) {
      addk(i).tempRowVec(rowChanged) = poke(i);
    });
  Kokkos::Profiling::popRegion();

  // 3. copyChangedRow for each walker 
  Kokkos::Profiling::pushRegion("updateRow::populateRcopy");
  Kokkos::parallel_for(Kokkos::MDRangePolicy<Kokkos::Rank<2,Kokkos::Iterate::Left> >({0,0},{numWalkers,numEls}),
		       KOKKOS_LAMBDA(const int& i0, const int& i1) {
			 addk(i0).rcopy(i1) = addk(i0).psiMinv(rowChanged,i1);
		       });
  Kokkos::Profiling::popRegion();

  // 4. do ger for each walker  
  Kokkos::Profiling::pushRegion("updateRow::ger");
  for (int i = 0; i < numWalkers; i++) {
    DiracDeterminant* ddp = static_cast<DiracDeterminant*>(WFC_list[i]);
    ddp->lah.ger(ddp->ddk.psiMinv, ddp->ddk.rcopy, ddk->ddk.tempRowVec, -cone);
  }
  Kokkos::Profiling::popRegion();

  // 5. copy the result back from psiV to the right row of psiMsave
  Kokkos::Profiling::pushRegion("copyBack");
  Kokkos::parallel_for(Kokkos::MDRangePolicy<Kokkos::Rank<2,Kokkos::Iterate::Left> >({0,0},{numWalkers,numEls}),
		       KOKKOS_LAMBDA(const int& i0, const int& i1) {
			 addk(i0).psiMsave(rowChanged,i1) = addk(i0).psiV(i1);
		       });
  Kokkos::Profiling::popRegion();
}

#ifdef KOKKOS_ENABLE_CUDA
// this is specialized to when the Views live on the GPU in CudaSpace 
template<typename addkType, typename vectorType>
void doDiracDeterminantMultiAccept<Kokkos::CudaSpace>(ddkType addk, vectorType& wfcv, int iel) {
  // for every walker, need to do updateRow followed by copyBack
  int numWalkers = WFC_list.size();
  int numEls = static_cast<DiracDeterminant*>(WFC_list[0])->psiV.extent(0);
  using ValueType = static_cast<DiracDeterminant*>(wfcv[0])->ddk.psiM::data_type;
  int rowChanged = iel-static_cast<DiracDeterminant*>(WFC_list[0]).FirstIndex;

  cudaStream_t *streams = (cudaStream_t *) malloc(addk.extent(0)*sizeof(cudaStream_t));
  for (int i = 0; i < numWalkers; i++) {
    cudaStreamCreate(&streams[i]);
  }

  // 1. gemvTrans
  Kokkos::Profiling::PushRegion("updateRow::gemvTrans");
  for (int i = 0; i < numWalkers; i++) {
    DiracDeterminant* ddp = static_cast<DiracDeterminant*>(WFC_list[i]);
    auto& lahref = ddp->lah;
    cublasSetStream(lahref.cublas_handle, streams[i]);

    DiracDeterminantKokkos& ddk = ddp->ddk;
    const ValueType c_ratio = cone / ddp->curRatio; 
    ddp->lah.gemvTrans(ddk.psiMinv, ddk.psiV, ddk.tempRowVec, c_ratio, czero); 
  }
  cudaDeviceSynchronize();
  Kokkos::Profiling::popRegion();
  
  // 2. poke one element on the device for each walker
  Kokkos::Profiling::pushRegion("updateRow::pokeSingleValue");
  Kokkos::View<ValueType*> poke("poke", numWalkers);
  auto pokeView = Kokkos:create_mirror_view(poke);
  for (int i = 0; i < numWalkers; i++) {
    DiracDeterminant* ddp = static_cast<DiracDeterminant>(WFC_list[i]);
    pokeView(i) = cone - cone / ddp->curRatio;
  }
  Kokkos::deep_copy(poke, pokeView);
  Kokkos::parallel_for(numWalkers, Kokkos_LAMBDA(int i) {
      addk(i).tempRowVec(rowChanged) = poke(i);
    });
  Kokkos::Profiling::popRegion();

  // 3. copyChangedRow for each walker 
  Kokkos::Profiling::pushRegion("updateRow::populateRcopy");
  Kokkos::parallel_for(Kokkos::MDRangePolicy<Kokkos::Rank<2,Kokkos::Iterate::Left> >({0,0},{numWalkers,numEls}),
		       KOKKOS_LAMBDA(const int& i0, const int& i1) {
			 addk(i0).rcopy(i1) = addk(i0).psiMinv(rowChanged,i1);
		       });
  Kokkos::Profiling::popRegion();

  // 4. do ger for each walker  
  Kokkos::Profiling::pushRegion("updateRow::ger");
  for (int i = 0; i < numWalkers; i++) {
    DiracDeterminant* ddp = static_cast<DiracDeterminant*>(WFC_list[i]);
    cublasSetStream(ddp->lah.cublas_handle, streams[i]);
    ddp->lah.ger(ddp->ddk.psiMinv, ddp->ddk.rcopy, ddk->ddk.tempRowVec, -cone);
  }
  cudaDeviceSynchronize();
  Kokkos::Profiling::popRegion();
  for (int i =0; i < numWalkers; i++) {
    cudaStreamDestroy(streams[i]);
  }
}
#endif
 
 
// this design is problematic because I cannot call cublas from inside of a device function
// probably will have to abandon this and fold it back into the earlier version, but perhaps I
// can keep the scalar values on the device (in views).  Then I would keep the structure of the 
// code as it is and just add functionality to the linalgHelper to work in a particular stream and
// also to be able to set cublas to expect the scalar arguments to be in the device memory

// can actually still keep this.  The point is that I can have a method in DiracDeterminantKokkos
// that takes a view full of DiracDeterminantKokkos and then I can operate on them directly.

// also, I can use the embedded linalghelpers to do the low level work, but I will certainly need
// to template these functions over the memory space.
 

// need to reorganize in two ways
// 1.  push data out into plain class that can live on the device
//      also add needed functionality inline here for convenience
struct DiracDeterminantKokkos : public QMCTraits
{
  using MatType = Kokkos::View<ValueType**, Kokkos::LayoutRight>;
  using DoubleMatType = Kokkos::View<double**, Kokkos::LayoutRight>;

  Kokkos::View<ValueType[1]> LogValue;
  Kokkos::View<ValueType[1]> curRatio;
  Kokkos::View<int[1]> FirstIndex;

  // inverse matrix to be updated
  MatType psiMinv;
  // storage for the row update 
  Kokkos::View<ValueType*> psiV;
  // temporary storage for row update
  Kokkos::View<ValueType*> tempRowVec;
  Kokkos::View<ValueType*> rcopy;
  // internal storage to perform inversion correctly
  MatType psiM;
  // temporary workspace for inversion
  MatType psiMsave;
  // temporary workspace for getrf
  Kokkos::View<ValueType*> getRfWorkSpace;
  Kokkos::View<ValueType**> getRiWorkSpace;
  // pivot array
  Kokkos::View<int*> piv;
  
  KOKKOS_INLINE_FUNCTION
  DiracDeterminantKokkos() { ; }

  KOKKOS_INLINE_FUNCTION
  DiracDeterminantKokkos* operator=(const DiracDeterminantKokkos& rhs) {
    LogValue = rhs.LogValue;
    curRatio = rhs.curRatio;
    FirstIndex = rhs.FirstIndex;
    psiMinv = rhs.psiMinv;
    psiV = rhs.psiV;
    tempRowVec = rhs.tempRowVec;
    rcopy = rhs.rcopy;
    psiMsave = rhs.psiMsave;
    getRfWorkSpace = rhs.getRfWorkSpace;
    getRiWorkSpace = rhs.getRiWorkSpace;
    piv = rhs.piv;
    return this;
  }

  DiracDeterminantKokkos(const DiracDeterminantKokkos&) = default;
  // need to add in checkMatrix(), evaluateLog(psk*), evalGrad(psk*, iat),
  //                ratioGrad(psk*, iat, gradType), evaluateGL(psk*, G, L, fromscratch)
  //                recompute(), ratio(psk*, iel), acceptMove(psk*, iel)
};


template<typename addkType, typename vectorType>
void populateCollectiveView(addkType addk, vectorType& WFC_list) {
  auto adkdMirror = Kokkos::create_mirror_view(addk);
  for (int i = 0; i < WFC_list.size(); i++) {
    addkMirror(i) = static_cast<DiracDeterminant*>(WFC_list[i])->ddk;
  }
  Kokkos::deep_copy(addk, addkMirror);
}

template<typename addkType, typename vectorType>
  void populateCollectiveView(addkType addk, vectorType& WFC_list, std::vector<bool>& isAccepted) {
  auto addkMirror = Kokkos::create_mirror_view(addk);

  int idx = 0;
  for (int i = 0; i < WFC_list.size(); i++) {
    if (isAccepted[i]) {
      addkMirror(idx) = static_cast<DiracDeterminant*>(WFC_list[i])->ddk;
      idx++;
    }
  }
  Kokkos::deep_copy(addk, addkMirror);
}
  

struct DiracDeterminant : public WaveFunctionComponent
{
  DiracDeterminant(int nels, const RandomGenerator<RealType>& RNG, int First = 0) 
    : FirstIndex(First), myRandom(RNG)
  {
    ddk.LogValue       = Kokkos::View<ValueType[1]>("LogValue");
    ddk.curRatio       = Kokkos::View<ValueType[1]>("curRatio");
    ddk.FirstIndex     = Kokkos::View<int[1]>("FirstIndex");
    ddk.psiMinv        = DiracDeterminantKokkos::MatType("psiMinv",nels,nels);
    ddk.psiM           = DiracDeterminantKokkos::MatType("psiM",nels,nels);
    ddk.psiMsave       = DiracDeterminantKokkos::MatType("psiMsave",nels,nels);
    ddk.psiV           = Kokkos::View<ValueType*>("psiV",nels);
    ddk.tempRowVec     = Kokkos::View<ValueType*>("tempRowVec", nels);
    ddk.rcopy          = Kokkos::View<ValueType*>("rcopy", nels);
    int getRfBufSize = getrf_gpu_buffer_size(nels, nels, ddk.psiM.data(), nels, lah);
    ddk.getRfWorkSpace = Kokkos::View<ValueType*>("getrfws", getRfBufSize);
    ddk.getRiWorkSpace = DiracDeterminantKokkos::MatType("getriws", nels, nels);
    ddk.piv            = Kokkos::View<int*>("piv", nels);
    
    FirstIndexMirror = Kokkos::create_mirror_view(ddk.FirstIndex);
    FirstIndexMirror(0) = FirstIndex;
    Kokkos::deep_copy(ddk.FirstIndex, FirstIndexMirror);
    
    // basically we are generating uniform random number for
    // each entry of psiMsave in the interval [-0.5, 0.5]
    constexpr double shift(0.5);

    // change this to match data ordering of DeterminantRef
    // recall that psiMsave has as its fast index the leftmost index
    // however int the c style matrix in DeterminantRef 
    auto psiMsaveMirror = Kokkos::create_mirror_view(psiMsave);
    auto psiMMirror = Kokkos::create_mirror_view(psiM);
    for (int i = 0; i < nels; i++) {
      for (int j = 0; j < nels; j++) {
	psiMsaveMirror(i,j) = myRandom.rand()-shift;
	psiMMirror(j,i) = psiMsaveMirror(i,j);
      }
    }
    Kokkos::deep_copy(psiMsave, psiMsaveMirror);
    Kokkos::deep_copy(psiM, psiMMirror);

    RealType phase;
    LogValue = InvertWithLog(ddk, lah, phase);
    elementWiseCopy(psiMinv, psiM);
  }

  void checkMatrix()
  {
    DiradDeterminantBase::MatType psiMRealType("psiM_RealType", ddk.psiM.extent(0), ddk.psiM.extent(0));
    elementWiseCopy(psiMRealType, ddk.psiM);
    checkIdentity(ddk.psiMsave, psiMRealType, "Psi_0 * psiM(T)", lah);
    checkIdentity(ddk.psiMsave, ddk.psiMinv, "Psi_0 * psiMinv(T)", lah);
    checkDiff(psiMRealType, ddk.psiMinv, "psiM - psiMinv(T)");
  }

  RealType evaluateLog(ParticleSet& P,
		       ParticleSet::ParticleGradient_t& G,
		       ParticleSet::ParticleLaplacian_t& L)
  {
    recompute();
    return 0.0;
  }

  GradType evalGrad(ParticleSet& P, int iat) { return GradType(); }
  ValueType ratioGrad(ParticleSet& P, int iat, GradType& grad) { return ratio(P, iat); }
  void evaluateGL(ParticleSet& P,
                  ParticleSet::ParticleGradient_t& G,
                  ParticleSet::ParticleLaplacian_t& L,
                  bool fromscratch = false) {}

  inline void recompute()
  {
    //elementWiseCopy(psiM, psiMsave); // needs to be transposed!
    elementWiseCopyTrans(ddk.psiM, ddk.psiMsave); // needs to be transposed!
    lah.invertMatrix(ddk.psiM);
    elementWiseCopy(ddk.psiMinv, ddk.psiM);
  }

  // in real application, inside here it would actually evaluate spos at the new
  // position and stick them in psiV
  inline ValueType ratio(ParticleSet& P, int iel)
  {
    const int nels = ddk.psiV.extent(0);
    constexpr double shift(0.5);
    //constexpr double czero(0);

    auto psiVMirror = Kokkos::create_mirror_view(psiV);
    for (int j = 0; j < nels; ++j) {
      psiVMirror(j) = myRandom() - shift;
    }
    Kokkos::deep_copy(ddk.psiV, psiVMirror);
    // in main line previous version this looked like:
    // curRatio = inner_product_n(psiV.data(), psiMinv[iel - FirstIndex], nels, czero);
    // same issues with indexing
    curRatio = lah.updateRatio(ddk.psiV, ddk.psiMinv, iel-FirstIndex);
    return curRatio;
  }
  inline void acceptMove(ParticleSet& P, int iel) {
    Kokkos::Profiling::pushRegion("Determinant::acceptMove");
    const int nels = ddk.psiV.extent(0);
    
    Kokkos::Profiling::pushRegion("Determinant::acceptMove::updateRow");
    updateRow(ddk.psiMinv, ddk.psiV, iel-FirstIndex, curRatio, lah);
    Kokkos::Profiling::popRegion();
    // in main line previous version this looked like:
    //std::copy_n(psiV.data(), nels, psiMsave[iel - FirstIndex]);
    // note 1: copy_n copies data from psiV to psiMsave
    //
    // note 2: the single argument call to psiMsave[] returned a pointer to
    // the iel-FirstIndex ROW of the underlying data structure, so
    // the operation was like (psiMsave.data() + (iel-FirstIndex)*D2)
    // note that then to iterate through the data it was going sequentially
    Kokkos::Profiling::pushRegion("Determinant::acceptMove::copyBack");
    lah.copyBack(ddk.psiMsave, ddk.psiV, iel-FirstIndex);

    Kokkos::Profiling::popRegion();
    Kokkos::Profiling::popRegion();
  }

  // accessor functions for checking
  inline double operator()(int i) const {
    // not sure what this was for, seems odd to 
    //Kokkos::deep_copy(psiMinv, psiMinv_host);
    int x = i / ddk.psiMinv_host.extent(0);
    int y = i % ddki.psiMinv_host.extent(0);
    auto dev_subview = subview(ddk.psiMinv, x, y);
    auto dev_subview_host = Kokkos::create_mirror_view(dev_subview);
    Kokkos::deep_copy(dev_subview_host, dev_subview);
    return dev_subview_host(0,0);
  }
  inline int size() const { return psiMinv.extent(0)*psiMinv.extent(1); }

  //// collective functions
  virtual void multi_evaluateLog(const std::vector<WaveFunctionComponent*>& WFC_list,
                                 const std::vector<ParticleSet*>& P_list,
                                 const std::vector<ParticleSet::ParticleGradient_t*>& G_list,
                                 const std::vector<ParticleSet::ParticleLaplacian_t*>& L_list,
                                 ParticleSet::ParticleValue_t& values) {

    Kokkos::View<DiracDeterminantKokkos*> addk("addk", WFC_list.size());
    populateCollectiveView(addk, WFC_list);
    
    // would just do it inline, but need to template on the memory space
    doDiracDeterminantMultiEvaluateLog<DiracDeterminantKokkos::MatType::memory_space>(ddk, WFC_list, values);    
  }

  // miniapp does nothing here, just return 0
  virtual void multi_evalGrad(const std::vector<WaveFunctionComponent*>& WFC_list,
			      const std::vector<ParticleSet*>& P_list,
                              int iat,
			      std::vector<posT>& grad_now) {
    for (int i = 0; i < grad_now.size(); i++) {
      grad_now(i) = posT();
    }
  }
  
  // do a loop over all walkers, stick ratio in ratios, do nothing to grad_new
  virtual void multi_ratioGrad(const std::vector<WaveFunctionComponent*>& WFC_list,
			       const std::vector<ParticleSet*>& P_list,
			       int iat,
			       std::vector<valT>& ratios,
			       std::vector<posT>& grad_new) {
    Kokkos::View<DiracDeterminantKokkos*> addk("addk", WFC_list.size());
    populateCollectiveView(addk, WFC_list);
    
    Kokkos::View<valT*> tempResults("tempResults", ratios.size());

    doDiracDeterminantMultiEvalRatio(addk, WFC_list, tempResults, iat);
    
    auto tempResultsMirror = Kokkos::create_mirror_view(tempResults);
    Kokkos::deep_copy(tempResultsMirror, tempResults);
    for (int i = 0; i < ratios.size(); i++) {
      ratios[i] = tempResultsMirror(i);
    }
  }

  virtual void multi_acceptRestoreMove(const std::vector<WaveFunctionComponent*>& WFC_list,
				       const std::vector<ParticleSet*>& P_list,
				       const std::vector<bool>& isAccepted,
				       int iat) {
    std::vector<WaveFunctionComponent*> activeWFC_list;
    for (int i = 0; i < WFC_list.size(); i++) {
      if (isAccepted(i)) {
	activeWFC_list.push_back(WFC_list(i));
      }
    }

    Kokkos::View<DiracDeterminantKokkos*> addk("addk", activeWFC_list.size());
    populateCollectiveView(addk, activeWFC_list);
	
    doDiracDeterminantMultiAccept<DiracDeterminantKokkos::MatType::memory_space>(addk, activeWFC_list, iat);
  }
    
  virtual void multi_evaluateGL(const std::vector<WaveFunctionComponent*>& WFC_list,
				const std::vector<ParticleSet*>& P_list,
				const std::vector<ParticleSet::ParticleGradient_t*>& G_list,
				const std::vector<ParticleSet::ParticleLaplacian_t*>& L_list,
				bool fromscratch = false) {
    // code in miniapp does NOTHING here
  }

  

  
  DiracDeterminantKokkos ddk;
  /// Helper class to handle linear algebra
  /// Holds for instance space for pivots and workspace
  linalgHelper<ValueType, DiracDeterminantKokkos::MatType::array_layout, DiracDeterminantKokkos::MatType::memory_space> lah;

  /// initial particle index
  const int FirstIndex;
  /// current ratio
  double curRatio;
  /// log|det|
  double LogValue;
  /// random number generator for testing
  RandomGenerator<RealType> myRandom;
private:

};
} // namespace qmcplusplus







#endif
