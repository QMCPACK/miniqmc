//////////////////////////////////////////////////////////////////////////////
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
#include "QMCWaveFunctions/DeterminantKokkos.h"
#include "QMCWaveFunctions/WaveFunctionKokkos.h"
#include "Utilities/RandomGenerator.h"


namespace qmcplusplus
{

struct DiracDeterminantKokkos;
struct DiracDeterminant;

template<class linAlgHelperType, typename ValueType>
ValueType InvertWithLog(DiracDeterminantKokkos& ddk, linAlgHelperType& lah, ValueType& phase);

template<class linAlgHelperType, typename ValueType>
void updateRow(DiracDeterminantKokkos& ddk, linAlgHelperType& lah, int rowchanged, ValueType c_ratio_in);

template<typename addkType, typename vectorType, typename resVecType>
void doDiracDeterminantMultiEvaluateLog(addkType& addk, vectorType& wfcv, resVecType& results);

template<typename addkType, typename vectorType, typename resVecType>
void doDiracDeterminantMultiEvaluateLog(addkType& addk, vectorType& wfcv, resVecType& results, const Kokkos::HostSpace&);

#ifdef KOKKOS_ENABLE_CUDA
template<typename addkType, typename vectorType, typename resVecType>
void doDiracDeterminantMultiEvaluateLog(addkType& addk, vectorType& wfcv, resVecType& results, const Kokkos::CudaSpace&);
template<typename addkType, typename vectorType, typename resVecType>
void doDiracDeterminantMultiEvaluateLog(addkType& addk, vectorType& wfcv, resVecType& results, const Kokkos::CudaUVMSpace&);
#endif

template<typename addkType, typename vectorType, typename resVecType>
void doDiracDeterminantMultiEvalRatio(addkType addk, vectorType& wfcv, resVecType& ratios, int iel);

template<typename addkType, typename vectorType, typename psiVsType, typename tempResType>
void doDiracDeterminantMultiEvalRatio(addkType& addk, vectorType& wfcv, psiVsType& psiVs,
				      tempResType ratios, Kokkos::View<int*>& isValidMap,
				      int numValid, int iel);

// using this one, 
template<typename eiListType, typename psiVType, typename ratiosType>
void doDiracDeterminantMultiEvalRatio(int pairNum, WaveFunctionKokkos& wfc, eiListType& eiList, 
				      psiVType& psiVScratch, ratiosType& ratios, int numActive);


template<typename addkType, typename vectorType>
void doDiracDeterminantMultiAccept(addkType& addk, vectorType& WFC_list, 
				   Kokkos::View<int*>& isAcceptedMap,
				   int numAccepted, int iel);
template<typename addkType, typename vectorType>
void doDiracDeterminantMultiAccept(addkType& addk, vectorType& WFC_list, 
				   Kokkos::View<int*>& isAcceptedMap,
				   int numAccepted, int iel, const Kokkos::HostSpace&);

#ifdef KOKKOS_ENABLE_CUDA
template<typename addkType, typename vectorType>
void doDiracDeterminantMultiAccept(addkType& addk, vectorType& WFC_list, 
				   Kokkos::View<int*>& isAcceptedMap,
				   int numAccepted, int iel, const Kokkos::CudaSpace&);
template<typename addkType, typename vectorType>
void doDiracDeterminantMultiAccept(addkType& addk, vectorType& WFC_list, 
				   Kokkos::View<int*>& isAcceptedMap,
				   int numAccepted, int iel, const Kokkos::CudaUVMSpace&);
#endif



template<typename addkType, typename vectorType>
void doDiracDeterminantMultiAccept(addkType& addk, vectorType& WFC_list, int iel);
template<typename addkType, typename vectorType>
void doDiracDeterminantMultiAccept(addkType& addk, vectorType& WFC_list, int iel, const Kokkos::HostSpace&);

#ifdef KOKKOS_ENABLE_CUDA
template<typename addkType, typename vectorType>
void doDiracDeterminantMultiAccept(addkType& addk, vectorType& WFC_list, int iel, const Kokkos::CudaSpace&);
template<typename addkType, typename vectorType>
void doDiracDeterminantMultiAccept(addkType& addk, vectorType& WFC_list, int iel, const Kokkos::CudaUVMSpace&);
#endif

template<typename addkType, typename vectorType>
void populateCollectiveView(addkType addk, vectorType& WFC_list);

template<typename addkType, typename vectorType>
void populateCollectiveView(addkType addk, vectorType& WFC_list, std::vector<bool>& isAccepted);




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
#ifdef KOKKOS_ENABLE_CUDA
    int getRfBufSize = getrf_gpu_buffer_size(nels, nels, ddk.psiM.data(), nels, lah.cusolver_handle);
    //std::cout << "in constructing DiracDeterminant, getRfBufSize = " << getRfBufSize << std::endl;
    ddk.getRfWorkSpace = Kokkos::View<ValueType*>("getrfws", getRfBufSize);
#endif
    ddk.getRiWorkSpace = Kokkos::View<ValueType**>("getriws", nels, nels);
    ddk.piv            = Kokkos::View<int*>("piv", nels);
    
    auto FirstIndexMirror = Kokkos::create_mirror_view(ddk.FirstIndex);
    FirstIndexMirror(0) = FirstIndex;
    //std::cout << "constructing determinant, firstIndex = " << FirstIndex << std::endl;
    Kokkos::deep_copy(ddk.FirstIndex, FirstIndexMirror);
    
    // basically we are generating uniform random number for
    // each entry of psiMsave in the interval [-0.5, 0.5]
    constexpr RealType shift(0.5);

    // change this to match data ordering of DeterminantRef
    // recall that psiMsave has as its fast index the leftmost index
    // however int the c style matrix in DeterminantRef 
    auto psiMsaveMirror = Kokkos::create_mirror_view(ddk.psiMsave);
    auto psiMMirror = Kokkos::create_mirror_view(ddk.psiM);
    for (int i = 0; i < nels; i++) {
      for (int j = 0; j < nels; j++) {
	psiMsaveMirror(i,j) = myRandom.rand()-shift;
	psiMMirror(j,i) = psiMsaveMirror(i,j);
      }
    }
    Kokkos::deep_copy(ddk.psiMsave, psiMsaveMirror);
    Kokkos::deep_copy(ddk.psiM, psiMMirror);

    RealType phase;
    LogValue = InvertWithLog(ddk, lah, phase);
    elementWiseCopy(ddk.psiMinv, ddk.psiM);
  }

  void checkMatrix()
  {
    DiracDeterminantKokkos::MatType psiMRealType("psiM_RealType", ddk.psiM.extent(0), ddk.psiM.extent(0));
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
    constexpr RealType shift(0.5);
    //constexpr RealType czero(0);

    auto psiVMirror = Kokkos::create_mirror_view(ddk.psiV);
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
    updateRow(ddk, lah, iel-FirstIndex, curRatio);
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
  inline RealType operator()(int i) const {
    // not sure what this was for, seems odd to 
    //Kokkos::deep_copy(psiMinv, psiMinv_host);
    int x = i / ddk.psiMinv.extent(0);
    int y = i % ddk.psiMinv.extent(0);
    auto dev_subview = subview(ddk.psiMinv, x, y);
    auto dev_subview_host = Kokkos::create_mirror_view(dev_subview);
    Kokkos::deep_copy(dev_subview_host, dev_subview);
    return dev_subview_host();
  }
  inline int size() const { return ddk.psiMinv.extent(0)*ddk.psiMinv.extent(1); }

  //// collective functions
  
  virtual void multi_evaluateLog(const std::vector<WaveFunctionComponent*>& WFC_list,
                                 const std::vector<ParticleSet*>& P_list,
                                 const std::vector<ParticleSet::ParticleGradient_t*>& G_list,
                                 const std::vector<ParticleSet::ParticleLaplacian_t*>& L_list,
                                 ParticleSet::ParticleValue_t& values) {
 
    if (WFC_list.size() > 0) {
      
      //std::cout << "in DiracDeterminant::multi_evaluateLog" << std::endl;
      Kokkos::View<DiracDeterminantKokkos*> addk("addk", WFC_list.size());
      populateCollectiveView(addk, WFC_list);
      //std::cout << "finished making collective views" << std::endl;
      
      //std::cout << "about to do diracDeterminantMultiEvaluateLog" << std::endl;
      // would just do it inline, but need to template on the memory space
      // need to choose here either the up or down determinants from wfc
      doDiracDeterminantMultiEvaluateLog(addk, WFC_list, values);    
      //std::cout << "finished diracDeterminantMultiEvaluateLog" << std::endl;
    }
  }

  // miniapp does nothing here, just return 0
  virtual void multi_evalGrad(const std::vector<WaveFunctionComponent*>& WFC_list,
			      const std::vector<ParticleSet*>& P_list,
                              int iat,
			      std::vector<PosType>& grad_now) {
    for (int i = 0; i < grad_now.size(); i++) {
      grad_now[i] = PosType();
    }
  }
  
  virtual void multi_evalGrad(const std::vector<WaveFunctionComponent*>& WFC_list,
                              WaveFunctionKokkos& wfc,
                              Kokkos::View<ParticleSet::pskType*> psk,
                              int iat,
                              std::vector<PosType>& grad_now) {
    for (int i = 0; i < grad_now.size(); i++) {
      grad_now[i] = PosType();
    }
  }

  
  // do a loop over all walkers, stick ratio in ratios, do nothing to grad_new
  virtual void multi_ratioGrad(const std::vector<WaveFunctionComponent*>& WFC_list,
			       const std::vector<ParticleSet*>& P_list,
			       int iat,
			       std::vector<ValueType>& ratios,
			       std::vector<PosType>& grad_new) {
    //std::cout << "      in DiracDeterminant multi_ratioGrad" << std::endl;
    if (WFC_list.size() > 0) {
      Kokkos::View<DiracDeterminantKokkos*> addk("addk", WFC_list.size());
      populateCollectiveView(addk, WFC_list);
      
      Kokkos::View<ValueType*> tempResults("tempResults", ratios.size());
      doDiracDeterminantMultiEvalRatio(addk, WFC_list, tempResults, iat);
      
      auto tempResultsMirror = Kokkos::create_mirror_view(tempResults);
      Kokkos::deep_copy(tempResultsMirror, tempResults);
      for (int i = 0; i < ratios.size(); i++) {
	ratios[i] = tempResultsMirror(i);
      }
    }
    //std::cout << "      finished DiracDeterminant multi_ratioGrad" << std::endl;
  }

  

  virtual void multi_acceptRestoreMove(const std::vector<WaveFunctionComponent*>& WFC_list,
				       const std::vector<ParticleSet*>& P_list,
				       const std::vector<bool>& isAccepted,
				       int iat) {
    std::vector<WaveFunctionComponent*> activeWFC_list;
    for (int i = 0; i < WFC_list.size(); i++) {
      if (isAccepted[i]) {
	activeWFC_list.push_back(WFC_list[i]);
      }
    }
    if (activeWFC_list.size() > 0) {
      
      Kokkos::View<DiracDeterminantKokkos*> addk("addk", activeWFC_list.size());
      populateCollectiveView(addk, activeWFC_list);
      
      doDiracDeterminantMultiAccept(addk, activeWFC_list, iat);
    }
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
  RealType curRatio;
  /// log|det|
  RealType LogValue;
  /// random number generator for testing
  RandomGenerator<RealType> myRandom;
private:

};



template<class linAlgHelperType, typename ValueType>
ValueType InvertWithLog(DiracDeterminantKokkos& ddk, linAlgHelperType& lah, ValueType& phase) {
  ValueType locLogDet(0.0);
  lah.getrf(ddk.psiM);
  auto locPiv = lah.getPivot(); // note, this is in device memory
  int sign_det = 1;

  Kokkos::parallel_reduce("dd-invertWithLog1", ddk.psiM.extent(0), KOKKOS_LAMBDA ( int i, int& cur_sign) {
      cur_sign = (locPiv(i) == i+1) ? 1 : -1;
      cur_sign *= (ddk.psiM(i,i) > 0) ? 1 : -1;
    }, Kokkos::Prod<int>(sign_det));

  Kokkos::parallel_reduce("dd-invertWithLog2", ddk.psiM.extent(0), KOKKOS_LAMBDA (int i, ValueType& v) {
      v += std::log(std::abs(ddk.psiM(i,i)));
    }, locLogDet);
  lah.getri(ddk.psiM);
  phase = (sign_det > 0) ? 0.0 : M_PI;
  //auto logValMirror = Kokkos::create_mirror_view(ddk.LogValue);
  //logValMirror(0) = locLogDet;
  //Kokkos::deep_copy(ddk.LogValue, logValMirror);
  return locLogDet;
}

template<class linAlgHelperType, typename ValueType>
void updateRow(DiracDeterminantKokkos& ddk, linAlgHelperType& lah, int rowchanged, ValueType c_ratio_in) {
  Kokkos::Profiling::pushRegion("dd-updateRow");
  constexpr ValueType cone(1.0);
  constexpr ValueType czero(0.0);
  ValueType c_ratio = cone / c_ratio_in;
  Kokkos::Profiling::pushRegion("updateRow::gemvTrans");
  lah.gemvTrans(ddk.psiMinv, ddk.psiV, ddk.tempRowVec, c_ratio, czero);
  Kokkos::Profiling::popRegion();

  // hard work to modify one element of temp on the device
  Kokkos::Profiling::pushRegion("updateRow::pokeSingleValue");
  auto devElem = subview(ddk.tempRowVec, rowchanged);
  auto devElem_mirror = Kokkos::create_mirror_view(devElem);
  devElem_mirror() = cone - c_ratio;
  Kokkos::deep_copy(devElem, devElem_mirror);
  Kokkos::Profiling::popRegion();

  Kokkos::Profiling::pushRegion("updateRow::populateRcopy");
  lah.copyChangedRow(rowchanged, ddk.psiMinv, ddk.rcopy);
  Kokkos::Profiling::popRegion();

  // now do ger
  Kokkos::Profiling::pushRegion("updateRow::ger");
  lah.ger(ddk.psiMinv, ddk.rcopy, ddk.tempRowVec, -cone);
  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::popRegion();
}

template<typename addkType, typename vectorType, typename resVecType>
void doDiracDeterminantMultiEvaluateLog(addkType& addk, vectorType& wfcv, resVecType& results) {
  Kokkos::Profiling::pushRegion("dd-MultiEvalLog-Outer");
  doDiracDeterminantMultiEvaluateLog(addk, wfcv, results, typename addkType::memory_space());
  Kokkos::Profiling::popRegion();
}


template<typename addkType, typename vectorType, typename resVecType>
void doDiracDeterminantMultiEvaluateLog(addkType& addk, vectorType& wfcv, resVecType& results, const Kokkos::HostSpace& ms) {
  // for each element in addk,
  //      1. copy transpose of psiMsave to psiM
  //      2. invert psiM
  //      3. copy psiM to psiMinv
  
  //std::cout << "in part 1" << std::endl;
  // 1. copy transpose of psiMsave to psiM for all walkers
  Kokkos::Profiling::pushRegion("dd-MultiEvalLog-copyallin");
  const int numWalkers = addk.extent(0);
  const int numEls = static_cast<DiracDeterminant*>(wfcv[0])->ddk.psiV.extent(0);
  //std::cout << "  numWalkers = " << numWalkers << ", numEls = " << numEls << std::endl;
  //std::cout << "  for walker 0, dimensions of psiM are: " << addk(0).psiM.extent(0) << " x " << addk(0).psiM.extent(1) << std::endl;
  //std::cout << "  for walker 0, dimensions of psiMsave are: " << addk(0).psiMsave.extent(0) << " x " << addk(0).psiMsave.extent(1) << std::endl;
  Kokkos::parallel_for("dd-elementWiseCopyTransAllPsiM",
		       Kokkos::MDRangePolicy<Kokkos::Rank<3,Kokkos::Iterate::Left> >({0,0,0}, {numWalkers, numEls, numEls}),
		       KOKKOS_LAMBDA(const int& i0, const int& i1, const int& i2) {
			 addk(i0).psiM(i1, i2) = addk(i0).psiMsave(i2,i1);
		       });
  Kokkos::fence();
  Kokkos::Profiling::popRegion();

  Kokkos::Profiling::pushRegion("dd-MultiEvalLog-invertMatrices");
  //std::cout << "in part 2" << std::endl;
  // 2. invert psiM.  This will loop over the walkers and invert each psiM.  Need to switch to a batched version of this
  // simplest thing would be to assume mkl and then have this be a kokkos parallel_for, inside of which you would call
  // mkl_set_num_threads_local before doing the linear algebra calls (wouldn't be able to use lah because couldn't follow the
  // pointer inside the kernel
  for (int i = 0; i < addk.extent(0); i++) {
    auto toInv = static_cast<DiracDeterminant*>(wfcv[i])->ddk.psiM;
    static_cast<DiracDeterminant*>(wfcv[i])->lah.invertMatrix(toInv);
  }
  Kokkos::Profiling::popRegion();

  //std::cout << "in part 3" << std::endl;
  // 3. copy psiM to psiMinv
  Kokkos::Profiling::pushRegion("dd-MultiEvalLog-copyback");
  Kokkos::parallel_for("dd-elementWiseCopyAllPsiM",
		       Kokkos::MDRangePolicy<Kokkos::Rank<3,Kokkos::Iterate::Left> >({0,0,0}, {numWalkers, numEls, numEls}),
		       KOKKOS_LAMBDA(const int& i0, const int& i1, const int& i2) {
			 addk(i0).psiMinv(i1, i2) = addk(i0).psiM(i1,i2);
		       });
  Kokkos::fence();
  Kokkos::Profiling::popRegion();

  for (int i = 0; i < results.size(); i++) {
    results[i] = 0.0;
  }
}


#ifdef KOKKOS_ENABLE_CUDA

template<typename addkType, typename vectorType, typename resVecType>
void dddMELGPU(addkType& addk, vectorType& wfcv, resVecType& results) {
  // for each element in addk,
  //      1. copy transpose of psiMsave to psiM
  //      2. invert psiM
  //      3. copy psiM to psiMinv

  using ValueType = DiracDeterminantKokkos::MatType::value_type;
  const int numWalkers = addk.extent(0);
  const int numEls = static_cast<DiracDeterminant*>(wfcv[0])->ddk.psiV.extent(0);

  Kokkos::Profiling::pushRegion("dd-MultiEvalLog-copyallin");  
  // 1. copy transpose of psiMsave to psiM for all walkers and also zero out temp matrices
  Kokkos::parallel_for("dd-elementWiseCopyTransAllPsiM",
		       Kokkos::MDRangePolicy<Kokkos::Rank<3,Kokkos::Iterate::Left> >({0,0,0}, {numWalkers, numEls, numEls}),
		       KOKKOS_LAMBDA(const int& i0, const int& i1, const int& i2) {
			 addk(i0).psiM(i1, i2) = addk(i0).psiMsave(i2,i1);
			 addk(i0).getRiWorkSpace(i1,i2) = 0.0;
		       });
  Kokkos::fence();
  Kokkos::Profiling::popRegion();

   // 2. invert psiM.  This will loop over the walkers and invert each psiM.  Need to switch to a batched version of this
  // simplest thing would be to assume mkl and then have this be a kokkos parallel_for, inside of which you would call
  // mkl_set_num_threads_local before doing the linear algebra calls

  Kokkos::Profiling::pushRegion("dd-MultiEvalLog-makeIdentity");  
  // set up temp spaces ahead of calls
  Kokkos::parallel_for("dd-makeIntoIdentity",
		       Kokkos::MDRangePolicy<Kokkos::Rank<2,Kokkos::Iterate::Left> >({0,0},{numWalkers,numEls}),
		       KOKKOS_LAMBDA(const int& i0, const int& i1) {
			 addk(i0).getRiWorkSpace(i1,i1) = 1.0;
		       });
  Kokkos::fence();
  Kokkos::Profiling::popRegion();

  Kokkos::Profiling::pushRegion("dd-MultiEvalLog-makestreams");  
  
  static std::vector<cudaStream_t> streams;
  static int numAllocatedStreams = 0;
  if (numAllocatedStreams < addk.extent(0)) {
    for (int i = numAllocatedStreams; i < addk.extent(0); i++) {
      cudaStream_t stream;
      cudaStreamCreate(&stream);
      streams.push_back(stream);
    }
    numAllocatedStreams = addk.extent(0);
  }


  //cudaStream_t *streams = (cudaStream_t *) malloc(addk.extent(0)*sizeof(cudaStream_t));
  //for (int i = 0; i < numWalkers; i++) {
  //  cudaStreamCreate(&streams[i]);
  //}  
  cudaDeviceSynchronize();
  Kokkos::Profiling::popRegion();

  Kokkos::Profiling::pushRegion("dd-MultiEvalLog-doInverse");  
  // unfortunately, need to access these through the vector anyway, cannot do addk(i).psiM on host!
  for (int i = 0; i < numWalkers; i++) {
    auto& lahref = static_cast<DiracDeterminant*>(wfcv[i])->lah;
    cusolverDnSetStream(lahref.cusolver_handle, streams[i]);
    auto& psiM = static_cast<DiracDeterminant*>(wfcv[i])->ddk.psiM;
 
    auto& getRfWs = static_cast<DiracDeterminant*>(wfcv[i])->ddk.getRfWorkSpace;
    auto& getRiWs = static_cast<DiracDeterminant*>(wfcv[i])->ddk.getRiWorkSpace;
    auto& piv = static_cast<DiracDeterminant*>(wfcv[i])->ddk.piv;

    getrf_gpu_impl(static_cast<int>(psiM.extent(0)), static_cast<int>(psiM.extent(1)),
		   lahref.pointerConverter(psiM.data()), static_cast<int>(psiM.extent(0)),
		   lahref.pointerConverter(getRfWs.data()),
		   piv.data(), lahref.info.data(), lahref.cusolver_handle);
    getri_gpu_impl(static_cast<int>(psiM.extent(0)), lahref.pointerConverter(psiM.data()), piv.data(),
		   lahref.pointerConverter(getRiWs.data()), lahref.info.data(), lahref.cusolver_handle);
  }
  cudaDeviceSynchronize();
  Kokkos::fence();  
  Kokkos::Profiling::popRegion();

  Kokkos::Profiling::pushRegion("dd-MultiEvalLog-destroystreams");  
  //for (int i =0; i < numWalkers; i++) {
  //  cudaStreamDestroy(streams[i]);
  // }
  Kokkos::Profiling::popRegion();

  Kokkos::Profiling::pushRegion("dd-MultiEvalLog-copyback");  
  // 3. copy getRiWs to psiM and to psiMinv
  Kokkos::parallel_for("dd-elementWiseCopyAllPsiM",
		       Kokkos::MDRangePolicy<Kokkos::Rank<3,Kokkos::Iterate::Left> >({0,0,0}, {addk.extent(0), addk(0).psiM.extent(0), addk(0).psiM.extent(1)}),
		       KOKKOS_LAMBDA(const int& i0, const int& i1, const int& i2) {
			 addk(i0).psiM(i1,i2) = addk(i0).getRiWorkSpace(i1,i2);
			 addk(i0).psiMinv(i1, i2) = addk(i0).getRiWorkSpace(i1,i2);
		       });
  Kokkos::fence();
  Kokkos::Profiling::popRegion();

  for (int i = 0; i < results.size(); i++) {
    results[i] = 0.0;
  }
}

template<typename addkType, typename vectorType, typename resVecType>
void doDiracDeterminantMultiEvaluateLog(addkType& addk, vectorType& wfcv, resVecType& results, const Kokkos::CudaSpace& ms) {
  dddMELGPU(addk, wfcv, results);
}

template<typename addkType, typename vectorType, typename resVecType>
void doDiracDeterminantMultiEvaluateLog(addkType& addk, vectorType& wfcv, resVecType& results, const Kokkos::CudaUVMSpace& ms) {
  dddMELGPU(addk, wfcv, results);
}

#endif

template<typename addkType, typename vectorType, typename psiVsType, typename tempResType>
void doDiracDeterminantMultiEvalRatio(addkType& addk, vectorType& wfcv, psiVsType& psiVs,
				      tempResType ratios, Kokkos::View<int*>& isValidMap,
				      int numValid, int iel) {
  using ValueType = typename tempResType::value_type;
  const int numWalkers = numValid;
  const int numEls = static_cast<DiracDeterminant*>(wfcv[0])->ddk.psiV.extent(0);

  // put values into psiV vectors
  Kokkos::parallel_for("dd-mevalRatio-fill-psiVs",
		       Kokkos::MDRangePolicy<Kokkos::Rank<2,Kokkos::Iterate::Left> >({0,0},{numWalkers,numEls}),
		       KOKKOS_LAMBDA(const int& walkIdx, const int& elNum) {
			 const int walkNum = isValidMap(walkIdx);
			 addk(walkNum).psiV(elNum) = psiVs(walkNum)(elNum);
		       });

  using BarePolicy = Kokkos::TeamPolicy<>;
  BarePolicy pol(numValid, 1, 32);
  Kokkos::parallel_for("dd-evalRatio-walker-loop", pol,
		       KOKKOS_LAMBDA(BarePolicy::member_type member) {
			 int walkerIdx = member.league_rank();
			 int walkerNum = isValidMap(walkerIdx);
			 const int idx = iel - addk(walkerNum).FirstIndex(0);
			 //std::cout << "walkerIdx = " << walkerIdx << ", walkerNum = " << walkerNum << ", idx = " << idx << std::endl;

			 ValueType sumOver = 0.0;
			 Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(member, numEls),
						 [=] (const int& i, ValueType& innersum) {
						   innersum += addk(walkerNum).psiV(i) * addk(walkerNum).psiMinv(idx,i); 
						 }, sumOver);
			 Kokkos::single(Kokkos::PerTeam(member), [=]() {
			     ratios(walkerNum) = sumOver;
			   });
		       });
}

 

template<typename addkType, typename vectorType, typename resVecType>
void doDiracDeterminantMultiEvalRatio(addkType addk, vectorType& wfcv, resVecType& ratios, int iel) {
  //std::cout << "    in doDiracDeterminantMultiEvalRatio" << std::endl;
  using ValueType = typename resVecType::value_type;
  //std::cout << "      using statement on ValueType is OK" << std::endl;
  const int numWalkers = addk.extent(0);
  //std::cout << "      setting the number of walkers from addk is OK" << std::endl;
  const int numEls = static_cast<DiracDeterminant*>(wfcv[0])->ddk.psiV.extent(0);
  //std::cout << "      grabbing number of electrons from vector or WaveFunctionComponent* is OK" << std::endl;

  constexpr typename DiracDeterminantKokkos::RealType shift(0.5);

  // could do this in parallel, but the random number stream wouldn't be the same.
  // could avoid the data transfer that way.  This is OK, because the real code would
  // be calling evaluateV on the sposet and that would happen on the device as well.
  //std::cout << "    about to put random numbers into psiV" << std::endl;
  for (int i = 0; i < numWalkers; i++) {
    auto& psiV = static_cast<DiracDeterminant*>(wfcv[i])->ddk.psiV;
    auto psiVMirror = Kokkos::create_mirror_view(psiV);
    for (int j = 0; j < numEls; j++) {
      psiVMirror(j) = static_cast<DiracDeterminant*>(wfcv[0])->myRandom() - shift;
    }
    Kokkos::deep_copy(psiV, psiVMirror);
  }
  //std::cout << "      finished putting random numbers into psiV" << std::endl;


  using BarePolicy = Kokkos::TeamPolicy<>;
  BarePolicy pol(numWalkers, 1, 32);
  Kokkos::parallel_for("dd-evalRatio-walker-loop", pol,
		       KOKKOS_LAMBDA(BarePolicy::member_type member) {
			 int walkerNum = member.league_rank();
			 ValueType sumOver = 0.0;
			 Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(member, numEls),
						 [=] (const int& i, ValueType& innersum) {
						   const int idx = iel - addk(walkerNum).FirstIndex(0);
						   innersum += addk(walkerNum).psiV(i) * addk(walkerNum).psiMinv(idx,i); 
						 }, sumOver);
			 Kokkos::single(Kokkos::PerTeam(member), [=]() {
			     ratios(walkerNum) = sumOver;
			   });
		       });
}


template<typename eiListType, typename psiVType, typename ratiosType>
void doDiracDeterminantMultiEvalRatio(int pairNum, WaveFunctionKokkos& wfc, eiListType& eiList, 
				      psiVType& psiVScratch, ratiosType& ratios, int numActive) {
  using ValueType = typename psiVType::value_type;
  const int numKnots = psiVScratch.extent(1);
  const int numWalkers = numActive;
  
  using BarePolicy = Kokkos::TeamPolicy<>;
  BarePolicy pol(numWalkers*numKnots, Kokkos::AUTO, 32);
  Kokkos::parallel_for("dd-evalRatio-general", pol,
		       KOKKOS_LAMBDA(BarePolicy::member_type member) {
			 const int walkerIdx = member.league_rank() / numKnots;
			 const int knotNum = member.league_rank() % numKnots;
			 const int walkerNum = wfc.activeMap(walkerIdx);
			 const int firstIndexInDD = wfc.activeDDs(walkerNum).FirstIndex(0);
			 const int bandIdx = eiList(walkerNum, pairNum, 0) - firstIndexInDD;
			 const int numElsInDD = wfc.activeDDs(walkerNum).psiMinv.extent(0);
			 Kokkos::parallel_reduce(Kokkos::TeamVectorRange(member, numElsInDD),
						 [=] (const int& i, ValueType& innersum) {
						   innersum += psiVScratch(walkerNum,knotNum,i+firstIndexInDD) *
						     wfc.activeDDs(walkerIdx).psiMinv(bandIdx,i);
						 }, wfc.knots_ratios_view(walkerIdx, knotNum));
		       });

  Kokkos::deep_copy(wfc.knots_ratios_view_mirror, wfc.knots_ratios_view);
  for (int i = 0; i < numWalkers; i++) {
    const int walkerNum = wfc.activeMapMirror(i);
    for (int j = 0; j < wfc.knots_ratios_view_mirror.extent(1); j++) {
      ratios[walkerNum*numKnots+j] = wfc.knots_ratios_view_mirror(i,j);
    }
  }
}

template<typename addkType, typename vectorType>
void doDiracDeterminantMultiAccept(addkType& addk, vectorType& WFC_list, 
				   Kokkos::View<int*>& isAcceptedMap,
				   int numAccepted, int iel) {
  doDiracDeterminantMultiAccept(addk, WFC_list, isAcceptedMap, numAccepted, iel, typename addkType::memory_space());
}

template<typename addkType, typename vectorType>
void doDiracDeterminantMultiAccept(addkType& addk, vectorType& WFC_list, 
				   Kokkos::View<int*>& isAcceptedMap, int numAccepted,
				   int iel, const Kokkos::HostSpace& ms) {
  Kokkos::Profiling::pushRegion("dd-MultiAccept");
  // for every walker, need to do updateRow followed by copyBack
  int numWalkers = numAccepted;
  int numEls = static_cast<DiracDeterminant*>(WFC_list[0])->ddk.psiV.extent(0);
  using ValueType = DiracDeterminantKokkos::MatType::value_type;
  int rowChanged = iel-static_cast<DiracDeterminant*>(WFC_list[0])->FirstIndex;
  constexpr ValueType cone(1.0);
  constexpr ValueType czero(0.0);

  auto isAcceptedMapMirror = Kokkos::create_mirror_view(isAcceptedMap);
  Kokkos::deep_copy(isAcceptedMapMirror, isAcceptedMap);

  // 1. gemvTrans
  Kokkos::Profiling::pushRegion("updateRow::gemvTrans");
  for (int i = 0; i < numAccepted; i++) {
    const int walkerNum = isAcceptedMapMirror(i);
    DiracDeterminant* ddp = static_cast<DiracDeterminant*>(WFC_list[walkerNum]);
    DiracDeterminantKokkos& ddk = ddp->ddk;
    const ValueType c_ratio = cone / ddp->curRatio; 
    ddp->lah.gemvTrans(ddk.psiMinv, ddk.psiV, ddk.tempRowVec, c_ratio, czero); 
  }
  Kokkos::Profiling::popRegion();

  // 2. poke one element on the device for each walker
  Kokkos::Profiling::pushRegion("updateRow::pokeSingleValue");
  Kokkos::View<ValueType*> poke("poke", numAccepted);
  auto pokeMirror = Kokkos::create_mirror_view(poke);
  for (int i = 0; i < numAccepted; i++) {
    const int walkerNum = isAcceptedMapMirror(i);
    DiracDeterminant* ddp = static_cast<DiracDeterminant*>(WFC_list[walkerNum]);
    pokeMirror(i) = cone - cone / ddp->curRatio;
  }
  Kokkos::deep_copy(poke, pokeMirror);
  Kokkos::parallel_for("dd-poking-values", numAccepted, KOKKOS_LAMBDA(int i) {
      const int walkerNum = isAcceptedMap(i);
      addk(walkerNum).tempRowVec(rowChanged) = poke(i);
    });
  Kokkos::Profiling::popRegion();

  // 3. copyChangedRow for each walker 
  Kokkos::Profiling::pushRegion("dd-updateRow::populateRcopy");
  Kokkos::parallel_for(Kokkos::MDRangePolicy<Kokkos::Rank<2,Kokkos::Iterate::Left> >({0,0},{numAccepted,numEls}),
		       KOKKOS_LAMBDA(const int& i0, const int& i1) {
			 const int walkerNum = isAcceptedMap(i0);
			 addk(walkerNum).rcopy(i1) = addk(walkerNum).psiMinv(rowChanged,i1);
		       });
  Kokkos::Profiling::popRegion();

    // 4. do ger for each walker  
  Kokkos::Profiling::pushRegion("updateRow::ger");
  for (int i = 0; i < numAccepted; i++) {
    const int walkerNum = isAcceptedMapMirror(i);
    DiracDeterminant* ddp = static_cast<DiracDeterminant*>(WFC_list[walkerNum]);
    ddp->lah.ger(ddp->ddk.psiMinv, ddp->ddk.rcopy, ddp->ddk.tempRowVec, -cone);
  }
  Kokkos::Profiling::popRegion();

  // 5. copy the result back from psiV to the right row of psiMsave
  Kokkos::Profiling::pushRegion("copyBack");
  Kokkos::parallel_for("dd-copyBack", Kokkos::MDRangePolicy<Kokkos::Rank<2,Kokkos::Iterate::Left> >({0,0},{numAccepted,numEls}),
		       KOKKOS_LAMBDA(const int& i0, const int& i1) {
			 const int walkerNum = isAcceptedMap(i0);
			 addk(walkerNum).psiMsave(rowChanged,i1) = addk(walkerNum).psiV(i1);
		       });
  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::popRegion();
}

#ifdef KOKKOS_ENABLE_CUDA
// this is specialized to when the Views live on the GPU in CudaSpace 
template<typename addkType, typename vectorType>
void dddMAGPU(addkType& addk, vectorType& wfcv, 
	      Kokkos::View<int*>& isAcceptedMap,
	      int numAccepted, int iel) {
  Kokkos::Profiling::pushRegion("dd-MultiAccept (GPU)");
  // for every walker, need to do updateRow followed by copyBack
  int numWalkers = numAccepted;
  int numEls = static_cast<DiracDeterminant*>(wfcv[0])->ddk.psiV.extent(0);
  using ValueType = DiracDeterminantKokkos::MatType::value_type;
  int rowChanged = iel-static_cast<DiracDeterminant*>(wfcv[0])->FirstIndex;
  constexpr ValueType cone(1.0);
  constexpr ValueType czero(0.0);

  auto isAcceptedMapMirror = Kokkos::create_mirror_view(isAcceptedMap);
  Kokkos::deep_copy(isAcceptedMapMirror, isAcceptedMap);

  static std::vector<cudaStream_t> streams;
  static int numAllocatedStreams = 0;
  if (numAllocatedStreams < numAccepted) {
    for (int i = numAllocatedStreams; i < numAccepted; i++) {
      cudaStream_t stream;
      cudaStreamCreate(&stream);
      streams.push_back(stream);
    }
    numAllocatedStreams = numAccepted;
  }
  

  /*
  cudaStream_t *streams = (cudaStream_t *) malloc(addk.extent(0)*sizeof(cudaStream_t));
  for (int i = 0; i < numAccepted; i++) {
    cudaStreamCreate(&streams[i]);
  }
  */

  // 1. gemvTrans
  Kokkos::Profiling::pushRegion("updateRow::gemvTrans");
  for (int i = 0; i < numAccepted; i++) {
    const int walkerNum = isAcceptedMapMirror(i);
    DiracDeterminant* ddp = static_cast<DiracDeterminant*>(wfcv[walkerNum]);
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
  Kokkos::View<ValueType*> poke("poke", numAccepted);
  auto pokeView = Kokkos::create_mirror_view(poke);
  for (int i = 0; i < numAccepted; i++) {
    const int walkerNum = isAcceptedMapMirror(i);
    DiracDeterminant* ddp = static_cast<DiracDeterminant*>(wfcv[walkerNum]);
    pokeView(i) = cone - cone / ddp->curRatio;
  }
  Kokkos::deep_copy(poke, pokeView);
  Kokkos::parallel_for("dd-pokeElement", numWalkers, KOKKOS_LAMBDA(int i) {
      const int walkerNum = isAcceptedMap(i);
      addk(walkerNum).tempRowVec(rowChanged) = poke(i);
    });
  Kokkos::Profiling::popRegion();

  // 3. copyChangedRow for each walker 
   Kokkos::Profiling::pushRegion("updateRow::populateRcopy");
  Kokkos::parallel_for("dd-populateRcopy", Kokkos::MDRangePolicy<Kokkos::Rank<2,Kokkos::Iterate::Left> >({0,0},{numAccepted,numEls}),
		       KOKKOS_LAMBDA(const int& i0, const int& i1) {
			 const int walkerNum = isAcceptedMap(i0);
			 addk(walkerNum).rcopy(i1) = addk(walkerNum).psiMinv(rowChanged,i1);
		       });
  Kokkos::Profiling::popRegion();

  // 4. do ger for each walker  
  Kokkos::Profiling::pushRegion("updateRow::ger");
  for (int i = 0; i < numAccepted; i++) {
    const int walkerNum = isAcceptedMapMirror(i);
    DiracDeterminant* ddp = static_cast<DiracDeterminant*>(wfcv[walkerNum]);
    cublasSetStream(ddp->lah.cublas_handle, streams[i]);
    ddp->lah.ger(ddp->ddk.psiMinv, ddp->ddk.rcopy, ddp->ddk.tempRowVec, -cone);
  }
  cudaDeviceSynchronize();


  /*
  using BarePolicy = Kokkos::TeamPolicy<>;
  BarePolicy pol(numEls*numAccepted, 32, 32);

  Kokkos::parallel_for("hand-rolled-multi-ger-withrcopy", pol,
		       KOKKOS_LAMBDA(BarePolicy::member_type member) {
			 const int walkerIdx = member.league_rank() / numEls;
			 const int walkerNum = isAcceptedMap(walkerIdx);
			 const int i = member.league_rank() % numEls;
			 //const ValueType temp = -cone * addk(walkerNum).rcopy(i);
			 const ValueType temp = -cone * addk(walkerNum).psiMinv(rowChanged,i);
			 Kokkos::parallel_for(Kokkos::TeamVectorRange(member, numEls),
					      [&] (const int& j) {
						addk(walkerNum).psiMinv(i,j) += temp * addk(walkerNum).tempRowVec(j);
					      });
		       });
  */						

  Kokkos::Profiling::popRegion();
  //for (int i =0; i < numWalkers; i++) {
  //  cudaStreamDestroy(streams[i]);
  //}
  Kokkos::Profiling::popRegion();
}

template<typename addkType, typename vectorType>
void doDiracDeterminantMultiAccept(addkType& addk, vectorType& WFC_list, 
				   Kokkos::View<int*>& isAcceptedMap,
				   int numAccepted, int iel, const Kokkos::CudaSpace& ms) {
  dddMAGPU(addk, WFC_list, isAcceptedMap, numAccepted, iel);
}

template<typename addkType, typename vectorType>
void doDiracDeterminantMultiAccept(addkType& addk, vectorType& WFC_list, 
				   Kokkos::View<int*>& isAcceptedMap,
				   int numAccepted, int iel, const Kokkos::CudaUVMSpace& ms) {
  dddMAGPU(addk, WFC_list, isAcceptedMap, numAccepted, iel);
}

#endif





  
template<typename addkType, typename vectorType>
void doDiracDeterminantMultiAccept(addkType& addk, vectorType& WFC_list, int iel) {
  doDiracDeterminantMultiAccept(addk, WFC_list, iel, typename addkType::memory_space());
}


template<typename addkType, typename vectorType>
void doDiracDeterminantMultiAccept(addkType& addk, vectorType& WFC_list, int iel, const Kokkos::HostSpace& ms) {
  Kokkos::Profiling::pushRegion("dd-MultiAccept");
  // for every walker, need to do updateRow followed by copyBack
  int numWalkers = WFC_list.size();
  int numEls = static_cast<DiracDeterminant*>(WFC_list[0])->ddk.psiV.extent(0);
  using ValueType = DiracDeterminantKokkos::MatType::value_type;
  int rowChanged = iel-static_cast<DiracDeterminant*>(WFC_list[0])->FirstIndex;
  constexpr ValueType cone(1.0);
  constexpr ValueType czero(0.0);

  // 1. gemvTrans
  Kokkos::Profiling::pushRegion("updateRow::gemvTrans");
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
  auto pokeView = Kokkos::create_mirror_view(poke);
  for (int i = 0; i < numWalkers; i++) {
    DiracDeterminant* ddp = static_cast<DiracDeterminant*>(WFC_list[i]);
    pokeView(i) = cone - cone / ddp->curRatio;
  }
  Kokkos::deep_copy(poke, pokeView);
  Kokkos::parallel_for("dd-pokeElement", numWalkers, KOKKOS_LAMBDA(int i) {
      addk(i).tempRowVec(rowChanged) = poke(i);
    });
  Kokkos::Profiling::popRegion();

  // 3. copyChangedRow for each walker 
  Kokkos::Profiling::pushRegion("updateRow::populateRcopy");
  Kokkos::parallel_for("dd-populateRcopy", Kokkos::MDRangePolicy<Kokkos::Rank<2,Kokkos::Iterate::Left> >({0,0},{numWalkers,numEls}),
		       KOKKOS_LAMBDA(const int& i0, const int& i1) {
			 addk(i0).rcopy(i1) = addk(i0).psiMinv(rowChanged,i1);
		       });
  Kokkos::Profiling::popRegion();

  // 4. do ger for each walker  
  Kokkos::Profiling::pushRegion("updateRow::ger");
  for (int i = 0; i < numWalkers; i++) {
    DiracDeterminant* ddp = static_cast<DiracDeterminant*>(WFC_list[i]);
    ddp->lah.ger(ddp->ddk.psiMinv, ddp->ddk.rcopy, ddp->ddk.tempRowVec, -cone);
  }
  Kokkos::Profiling::popRegion();

  // 5. copy the result back from psiV to the right row of psiMsave
  Kokkos::Profiling::pushRegion("copyBack");
  Kokkos::parallel_for("dd-copyback", Kokkos::MDRangePolicy<Kokkos::Rank<2,Kokkos::Iterate::Left> >({0,0},{numWalkers,numEls}),
		       KOKKOS_LAMBDA(const int& i0, const int& i1) {
			 addk(i0).psiMsave(rowChanged,i1) = addk(i0).psiV(i1);
		       });
  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::popRegion();
}

#ifdef KOKKOS_ENABLE_CUDA
// this is specialized to when the Views live on the GPU in CudaSpace 
template<typename addkType, typename vectorType>
void dddMAGPU(addkType& addk, vectorType& wfcv, int iel) {
  Kokkos::Profiling::pushRegion("dd-MultiAccept (GPU)");
  // for every walker, need to do updateRow followed by copyBack
  int numWalkers = wfcv.size();
  int numEls = static_cast<DiracDeterminant*>(wfcv[0])->ddk.psiV.extent(0);
  using ValueType = DiracDeterminantKokkos::MatType::value_type;
  int rowChanged = iel-static_cast<DiracDeterminant*>(wfcv[0])->FirstIndex;
  constexpr ValueType cone(1.0);
  constexpr ValueType czero(0.0);

  static std::vector<cudaStream_t> streams;
  static int numAllocatedStreams = 0;
  if (numAllocatedStreams < addk.extent(0)) {
    for (int i = numAllocatedStreams; i < addk.extent(0); i++) {
      cudaStream_t stream;
      cudaStreamCreate(&stream);
      streams.push_back(stream);
    }
    numAllocatedStreams = addk.extent(0);
  }


  //cudaStream_t *streams = (cudaStream_t *) malloc(addk.extent(0)*sizeof(cudaStream_t));
  //for (int i = 0; i < numWalkers; i++) {
  //  cudaStreamCreate(&streams[i]);
  //}

  // 1. gemvTrans
  Kokkos::Profiling::pushRegion("updateRow::gemvTrans");
  for (int i = 0; i < numWalkers; i++) {
    DiracDeterminant* ddp = static_cast<DiracDeterminant*>(wfcv[i]);
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
  auto pokeView = Kokkos::create_mirror_view(poke);
  for (int i = 0; i < numWalkers; i++) {
    DiracDeterminant* ddp = static_cast<DiracDeterminant*>(wfcv[i]);
    pokeView(i) = cone - cone / ddp->curRatio;
  }
  Kokkos::deep_copy(poke, pokeView);
  Kokkos::parallel_for("dd-pokeSingleValue", numWalkers, KOKKOS_LAMBDA(int i) {
      addk(i).tempRowVec(rowChanged) = poke(i);
    });
  Kokkos::Profiling::popRegion();

  // 3. copyChangedRow for each walker 
  Kokkos::Profiling::pushRegion("updateRow::populateRcopy");
  Kokkos::parallel_for("dd-populateRcopy", Kokkos::MDRangePolicy<Kokkos::Rank<2,Kokkos::Iterate::Left> >({0,0},{numWalkers,numEls}),
		       KOKKOS_LAMBDA(const int& i0, const int& i1) {
			 addk(i0).rcopy(i1) = addk(i0).psiMinv(rowChanged,i1);
		       });
  Kokkos::Profiling::popRegion();

  // 4. do ger for each walker  
  Kokkos::Profiling::pushRegion("updateRow::ger");
  for (int i = 0; i < numWalkers; i++) {
    DiracDeterminant* ddp = static_cast<DiracDeterminant*>(wfcv[i]);
    cublasSetStream(ddp->lah.cublas_handle, streams[i]);
    ddp->lah.ger(ddp->ddk.psiMinv, ddp->ddk.rcopy, ddp->ddk.tempRowVec, -cone);
  }
  cudaDeviceSynchronize();
  Kokkos::Profiling::popRegion();
  //for (int i =0; i < numWalkers; i++) {
  //  cudaStreamDestroy(streams[i]);
  //}
  Kokkos::Profiling::popRegion();
}

template<typename addkType, typename vectorType>
void doDiracDeterminantMultiAccept(addkType& addk, vectorType& wfcv, int iel, const Kokkos::CudaSpace& ms) {
  dddMAGPU(addk, wfcv, iel);
}
template<typename addkType, typename vectorType>
void doDiracDeterminantMultiAccept(addkType& addk, vectorType& wfcv, int iel, const Kokkos::CudaUVMSpace& ms) {
  dddMAGPU(addk, wfcv, iel);
}
#endif
 
 


template<typename addkType, typename vectorType>
void populateCollectiveView(addkType addk, vectorType& WFC_list) {
  auto addkMirror = Kokkos::create_mirror_view(addk);
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
  

} // namespace qmcplusplus







#endif
