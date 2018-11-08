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

#ifndef QMCPLUSPLUS_DETERMINANT_DEVICE_H
#define QMCPLUSPLUS_DETERMINANT_DEVICE_H

#include <Kokkos_Core.hpp>
#include <impl/Kokkos_Timer.hpp>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#ifdef KOKKOS_ENABLE_CUDA
#include "cublas_v2.h"
#include "cusolverDn.h"
#endif
#include "Utilities/Configuration.h"
#include "QMCWaveFunctions/WaveFunctionComponent.h"
//#include "Utilities/RandomGenerator.h"
namespace qmcplusplus
{
namespace future
{

template<class DEVICETYPE>
class DeterminantDevice
{
public:
  using QMCT = QMCTraits;

  DeterminantDevice(int nels, const RandomGenerator<QMCT::RealType>& RNG,
		    int First = 0) {}
  void checkMatrix()
  {
    static_cast<DEVICETYPE*>(this)->checkMatrixImp();
  }

  QMCT::RealType evaluateLog(ParticleSet& P,
		       ParticleSet::ParticleGradient_t& G,
		       ParticleSet::ParticleLaplacian_t& L)
  {
    return static_cast<DEVICETYPE*>(this)->evaluateLogImp(P, G, L);
  }

  QMCT::GradType evalGrad(ParticleSet& P, int iat)
  {
    return static_cast<DEVICETYPE*>(this)->evalGradImp(P, iat);
  }
  
  QMCT::ValueType ratioGrad(ParticleSet& P, int iat, QMCT::GradType& grad)
  {
    return static_cast<DEVICETYPE*>(this)->ratioImp(P, iat);
  }
  
  void evaluateGL(ParticleSet& P,
                  ParticleSet::ParticleGradient_t& G,
                  ParticleSet::ParticleLaplacian_t& L,
                  bool fromscratch = false)
  {
    static_cast<DEVICETYPE*>(this)->evaluateGLImp(P, G, L, fromscratch);
  }
  
  inline void recompute()
  {
    static_cast<DEVICETYPE*>(this)->recomputeImp();
  }
  
  inline QMCT::ValueType ratio(ParticleSet& P, int iel)
  {
    return static_cast<DEVICETYPE*>(this)->ratioImp(P, iel);
  }
  
  inline void acceptMove(ParticleSet& P, int iel) {
    static_cast<DEVICETYPE*>(this)->acceptMoveImp(P, iel);
  }

  /** accessor functions for checking?
   *
   *  would like to have const on here but with CRTP...
   */
  inline double operator()(int i) 
  {
    return static_cast<DEVICETYPE*>(this)->operatorParImp(i);
  }

  inline int size() const { return static_cast<DEVICETYPE*>(this)->sizeImp(); }
};
} // namespace future
} // namespace qmcplusplus

#endif
