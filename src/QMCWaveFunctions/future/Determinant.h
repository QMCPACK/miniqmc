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

#ifndef QMCPLUSPLUS_FUTURE_DETERMINANT_H
#define QMCPLUSPLUS_FUTURE_DETERMINANT_H

#include <Kokkos_Core.hpp>
#include <impl/Kokkos_Timer.hpp>
#include <cstdio>
#include <cstdlib>
#include <type_traits>

#include "QMCWaveFunctions/WaveFunctionComponent.h"
#include "QMCWaveFunctions/future/DeterminantDevice.h"
#include "QMCWaveFunctions/future/DeterminantDeviceImpKOKKOS.h"
//#include "Utilities/RandomGenerator.h"

namespace qmcplusplus
{
namespace future
{

template<class DEVICE>
struct DiracDeterminant : public WaveFunctionComponent
{
  DiracDeterminant(int nels, const RandomGenerator<RealType>& RNG, int First = 0) 
    : determinant_device<DEVICE>(nels, RNG, First), FirstIndex(First), myRandom(RNG)
  {
  }
  
  void checkMatrix()
  {
    determinant_device.checkMatrix();
  }
  
  RealType evaluateLog(ParticleSet& P,
		       ParticleSet::ParticleGradient_t& G,
		       ParticleSet::ParticleLaplacian_t& L)
  {
    return determinant_device.recompute(P, G, L);
  }

  GradType evalGrad(ParticleSet& P, int iat)
  {
    return determinant_device.GradType(P, iat);
  }
  ValueType ratioGrad(ParticleSet& P, int iat, GradType& grad)
  {
    return determinant_device.ratioGrad(P, iat, grad);
  }

  void evaluateGL(ParticleSet& P,
                  ParticleSet::ParticleGradient_t& G,
                  ParticleSet::ParticleLaplacian_t& L,
                  bool fromscratch = false)
  {
    determinant_device.ratioGrad(P, G, L, fromscratch);
  }

  inline void recompute()
  {
    determinant_device.recompute();
  }
  
  inline ValueType ratio(ParticleSet& P, int iel)
  {
    return determinant_device.ratio(P, iel);
  }
  
  inline void acceptMove(ParticleSet& P, int iel) {
    determinant_device.acceptMove(P, iel);
  }

  // accessor functions for checking
  inline double operator()(int i) const {
    return determinant_device.operator()(i);
  }
  
  inline int size() const { determinant_device.size(); }

private:

};

} // namespace future
} // namespace qmcplusplus

#endif
