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

template<class ViewType, class LinAlgHelperType, typename value_type>
value_type InvertWithLog(ViewType view, LinAlgHelperType& lah, value_type& phase) {
  value_type logdet(0.0);
  lah.getrf(view);
  auto locPiv = lah.getPivot(); // note this is in device memory!
  int sign_det = 1;
  
  Kokkos::parallel_reduce(view.extent(0), KOKKOS_LAMBDA ( int ii, int& cur_sign) {
      cur_sign = (locPiv(i) == i+1) ? 1 : -1;
      cur_sign *= (view(i,i) > 0) ? 1 : -1;
    }, Kokkos::Prod<int>(sign_det));

  Kokkos::parallel_reduce(view.extent(0), KOKKOS_LAMBDA (int ii, value_type& v) {
      v += std::log(std::abs(view(i,i)));
    }, logdet);

  lah.getri(view);
  phase = (sign_det > 0) ? 0.0 : M_PI;
  return logdet;
}

template<class ViewType, class ArrayViewType, class LinAlgHelperType, typename value_type>
void updateRow(ViewType pinv, ArrayViewType tv, int rowchanged, value_type c_ratio_in, LinAlgHelperType& lah) {
  constexpr value_type cone(1.0);
  constexpr value_type czero(0.0);
  ArrayViewType temp("temp", tv.extent(0));
  ArrayViewType rcopy("rcopy", tv.extent(0));
  value_type c_ratio = cone / c_ratio_in;
  Kokkos::Profiling::pushRegion("updateRow::gemvTrans");
  lah.gemvTrans(pinv, tv, temp, c_ratio, czero);
  Kokkos::Profiling::popRegion();

  // hard work to modify one element of temp on the device
  Kokkos::Profiling::pushRegion("updateRow::pokeSingleValue");
  auto devElem = subview(temp, rowchanged);
  auto devElem_mirror = Kokkos::create_mirror_view(devElem);
  devElem_mirror(0) = cone - c_ratio;
  Kokkos::deep_copy(devElem, devElem_mirror);
  Kokkos::Profiling::popRegion();

  // now extract the proper part of pinv into rcopy
  // in previous version this looked like: std::copy_n(pinv + m * rowchanged, m, rcopy);
  // a little concerned about getting the ordering wrong
  Kokkos::Profiling::pushRegion("updateRow::populateRcopy");

  lah.copyChangedRow(rowchanged, pinv, rcopy);
  Kokkos::Profiling::popRegion();
      
  // now do ger
  Kokkos::Profiling::pushRegion("updateRow::ger");
  lah.ger(pinv, rcopy, temp, -cone);
  Kokkos::Profiling::popRegion();
}



// need to reorganize in two ways
// 1.  push data out into plain class that can live on the device
//      also add needed functionality inline here for convenience
// 2.  fix linalgHelper so that its methods can be called from within a kernel
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
  DoubleMatType psiM;
  // temporary workspace for inversion
  MatType psiMsave;

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
    return this;
  }

  DiracDeterminantKokkos(const DiracDeterminantKokkos&) = default;

  template<class linAlgHelperType>
  KOKKOS_INLINE_FUNCTION
  ValueType InvertWithLog(LinAlgHelperType& lah, ValueType& phase) {
    ValueType locLogDet(0.0);
    lah.getrf(psiM);
    auto locPiv = lah.getPivot(); // needs to return a view whose memory is on the device
    int sign_det = 1;

    for (int i = 0; i < psiM.extent(0); i++) {
      sign_det *= (locPiv(i) == i+1) ? 1 : -1;
      sign_det *= (psiM(i,i) > 0) ? 1 : -1;
      LocLogDet += std::log(std::abs(psiM(i,i)));
    }
    lah.getri(psiM);
    phase = (sign_det > 0) ? 0.0 : M_PI;
    LogValue(0) = locLogDet;
    return locLogDet;
  }

  // called with psiMinv as first arg and psiV as the second
  template<class linAlgHelperType>
  KOKKOS_INLINE_FUNCTION
  void updateRow(int rowchanged, ValueType c_ratio_in, linAlgHelperType& lah) {
    constexpr ValueType cone(1.0);
    constexpr ValueType czero(0.0);
    ValueType c_ratio = cone / c_ratio_in;
    Kpkkos::Profiling::PushRegion("updateRow::gemvTrans");
    lah.gemvTrans(psiMinv, psiV, tempRowVec, c_ratio, czero);
    Kokkos::Profiling::popRegion();

    // this was a lot harder when we weren't inside a kernel already
    tempRowVec(rowchanged) = cone - c_ratio;

    Kokkos::Profiling::pushRegion("updateRow::populateRcopy");
    lah.copyChangedRow(rowchanged, psiMinv, rcopy);
    Kokkos::Profiling::popRegion();

    // now do ger
    Kokkos::Profiling::pushRegion("updateRow::ger");
    lah.ger(psiMinv, rcopy, tempRowVec, -cone);
    Kokkos::Profiling::popRegion();
  }

  // need to add in checkMatrix(), evaluateLog(psk*), evalGrad(psk*, iat),
  //                ratioGrad(psk*, iat, gradType), evaluateGL(psk*, G, L, fromscratch)
  //                recompute(), ratio(psk*, iel), acceptMove(psk*, iel)

  
};

  

 
			 

      

struct DiracDeterminant : public WaveFunctionComponent
{
  DiracDeterminant(int nels, const RandomGenerator<RealType>& RNG, int First = 0) 
    : FirstIndex(First), myRandom(RNG)
  {
    psiMinv = DoubleMatType("psiMinv",nels,nels);
    psiM = DoubleMatType("psiM",nels,nels);
    psiMsave = DoubleMatType("psiMsave",nels,nels);
    psiV = Kokkos::View<RealType*>("psiV",nels);

    psiMinv_host = Kokkos::create_mirror_view(psiMinv);
    psiMsave_host = Kokkos::create_mirror_view(psiMsave);
    psiM_host = Kokkos::create_mirror_view(psiM);
    psiV_host = Kokkos::create_mirror_view(psiV);

    // basically we are generating uniform random number for
    // each entry of psiMsave in the interval [-0.5, 0.5]
    constexpr double shift(0.5);

    // change this to match data ordering of DeterminantRef
    // recall that psiMsave has as its fast index the leftmost index
    // however int the c style matrix in DeterminantRef 
    for (int i = 0; i < nels; i++) {
      for (int j = 0; j < nels; j++) {
	psiMsave_host(i,j) = myRandom.rand()-shift;
      }
    }
    Kokkos::deep_copy(psiMsave, psiMsave_host);
     
    RealType phase;
    
    for (int i = 0; i < nels; i++) {
      for (int j = 0; j < nels; j++) {
	psiM_host(i,j) = psiMsave_host(j,i);
      }
    }
    Kokkos::deep_copy(psiM, psiM_host);

    LogValue = InvertWithLog(psiM, lah, phase);
    elementWiseCopy(psiMinv, psiM);
  }
  void checkMatrix()
  {
    MatType psiMRealType("psiM_RealType", psiM.extent(0), psiM.extent(0));
    elementWiseCopy(psiMRealType, psiM);
    checkIdentity(psiMsave, psiMRealType, "Psi_0 * psiM(T)", lah);
    checkIdentity(psiMsave, psiMinv, "Psi_0 * psiMinv(T)", lah);
    checkDiff(psiMRealType, psiMinv, "psiM - psiMinv(T)");
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
                  bool fromscratch = false)
  {}
  inline void recompute()
  {
    //elementWiseCopy(psiM, psiMsave); // needs to be transposed!
    elementWiseCopyTrans(psiM, psiMsave); // needs to be transposed!
    lah.invertMatrix(psiM);
    elementWiseCopy(psiMinv, psiM);
  }
  inline ValueType ratio(ParticleSet& P, int iel)
  {
    const int nels = psiV.extent(0);
    constexpr double shift(0.5);
    //constexpr double czero(0);
    for (int j = 0; j < nels; ++j) {
      psiV_host(j) = myRandom() - shift;
    }
    Kokkos::deep_copy(psiV, psiV_host);
    // in main line previous version this looked like:
    // curRatio = inner_product_n(psiV.data(), psiMinv[iel - FirstIndex], nels, czero);
    // same issues with indexing
    curRatio = lah.updateRatio(psiV, psiMinv, iel-FirstIndex);

    return curRatio;
  }
  inline void acceptMove(ParticleSet& P, int iel) {
    Kokkos::Profiling::pushRegion("Determinant::acceptMove");
    const int nels = psiV.extent(0);
    
    Kokkos::Profiling::pushRegion("Determinant::acceptMove::updateRow");
    updateRow(psiMinv, psiV, iel-FirstIndex, curRatio, lah);
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
    lah.copyBack(psiMsave, psiV, iel-FirstIndex);

    /*
    const int FirstIndex_ = FirstIndex;
    Kokkos::View<RealType*> psiV_=psiV;
    const int iel_=iel;
    DoubleMatType psiMsave_=psiMsave; 



    Kokkos::parallel_for( nels, KOKKOS_LAMBDA (int i) {
    	psiMsave_(iel_-FirstIndex_, i) = psiV_(i);
      });
    Kokkos::fence();
    */
    Kokkos::Profiling::popRegion();
    Kokkos::Profiling::popRegion();
  }

  // accessor functions for checking
  inline double operator()(int i) const {
    // not sure what this was for, seems odd to 
    //Kokkos::deep_copy(psiMinv, psiMinv_host);
    int x = i / psiMinv_host.extent(0);
    int y = i % psiMinv_host.extent(0);
    auto dev_subview = subview(psiMinv, x, y);
    auto dev_subview_host = Kokkos::create_mirror_view(dev_subview);
    Kokkos::deep_copy(dev_subview_host, dev_subview);
    return dev_subview_host(0,0);
  }
  inline int size() const { return psiMinv.extent(0)*psiMinv.extent(1); }

private:
  /// log|det|
  double LogValue;
  /// current ratio
  double curRatio;
  /// initial particle index
  const int FirstIndex;
  /// matrix type and mirror type
  //using MatType = Kokkos::View<RealType**, Kokkos::LayoutLeft>;
  using MatType = Kokkos::View<RealType**, Kokkos::LayoutRight>;
  using MatMirrorType = MatType::HostMirror;
  //using DoubleMatType = Kokkos::View<double**, Kokkos::LayoutLeft>;
  using DoubleMatType = Kokkos::View<double**, Kokkos::LayoutRight>;
  using DoubleMatMirrorType = DoubleMatType::HostMirror;
  /// inverse matrix to be updated and host mirror (kept in double regardless of RealType)
  MatType psiMinv;
  MatMirrorType psiMinv_host;
  /// storage for the row update and host mirror
  Kokkos::View<RealType*> psiV;
  Kokkos::View<RealType*>::HostMirror psiV_host;
  /// internal storage to perform inversion correctly and host mirror
  DoubleMatType psiM;
  DoubleMatMirrorType psiM_host;
  /// temporary workspace for inversion and host mirror
  MatType psiMsave;
  MatMirrorType psiMsave_host;
  /// random number generator for testing
  RandomGenerator<RealType> myRandom;
  /// Helper class to handle linear algebra
  /// Holds for instance space for pivots and workspace
  linalgHelper<MatType::value_type, MatType::array_layout, MatType::memory_space> lah;
};
} // namespace qmcplusplus







#endif
