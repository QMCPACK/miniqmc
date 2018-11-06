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

#include "QMCWaveFunctions/WaveFunctionComponent.h"
//#include "Utilities/RandomGenerator.h"
namespace qmcplusplus
{
namespace future
{

template<class DEVICETYPE>
class DeterminantDevice
{
  void checkMatrix()
  {
    static_cast<DEVICETYPE*>(this)->checkMatrix();
  }

  RealType evaluateLog(ParticleSet& P,
		       ParticleSet::ParticleGradient_t& G,
		       ParticleSet::ParticleLaplacian_t& L)
  {
    return static_cast<DEVICETYPE*>(this)->evaluateLog(P, G, L);
  }

  GradType evalGrad(ParticleSet& P, int iat)
  {
    return static_cast<DEVICETYPE*>(this)->GradType(P, iat);
  }
  
  ValueType ratioGrad(ParticleSet& P, int iat, GradType& grad)
  {
    return static_cast<DEVICETYPE*>(this)->ratio(P, iat);
  }
  
  void evaluateGL(ParticleSet& P,
                  ParticleSet::ParticleGradient_t& G,
                  ParticleSet::ParticleLaplacian_t& L,
                  bool fromscratch = false)
  {
    static_cast<DEVICETYPE*>(this)->evaluateGL(P, G, L, fromscratch);
  }
  
  inline void recompute()
  {
    static_cast<DEVICETYPE*>(this)->recompute();
  }
  
  inline ValueType ratio(ParticleSet& P, int iel)
  {
    return static_cast<DEVICETYPE*>(this)->ratio(P, iel);
  }
  
  inline void acceptMove(ParticleSet& P, int iel) {
    static_cast<DEVICETYPE*>(this)->acceptMove(P, iel);
  }

  // accessor functions for checking
  inline double operator()(int i) const
  {
    return static_cast<DEVICETYPE*>(this)->operator()(i);
  }

  inline int size() const { return static_cast<DEVICETYPE*>(this)->size(); }
};
} // namespace future
} // namespace qmcplusplus

#endif
