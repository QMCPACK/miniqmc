////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2018 QMCPACK developers.
//
// File developed by: Peter Doak, doakpw@ornl.gov, Oak Ridge National Lab
//                    M. Graham Lopez
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

#include "clean_inlining.h"
#include <cstdio>
#include <cstdlib>
#include <type_traits>

#include "QMCWaveFunctions/WaveFunctionComponent.h"
#include "QMCWaveFunctions/DeterminantDevice.h"
 
namespace qmcplusplus
{

template<class DEVICE>
struct DiracDeterminant : public WaveFunctionComponent
{
  DiracDeterminant(int nels, const RandomGenerator<RealType>& RNG, int First = 0)
  {
    determinant_device = new DEVICE(nels, RNG, First);
  }

  ~DiracDeterminant()
  {
    delete determinant_device;
  }
  
  void checkMatrix()
  {
    determinant_device->checkMatrix();
  }
  
  RealType evaluateLog(ParticleSet& P,
		       ParticleSet::ParticleGradient_t& G,
		       ParticleSet::ParticleLaplacian_t& L)
  {
    return determinant_device->evaluateLog(P, G, L);
  }

  GradType evalGrad(ParticleSet& P, int iat)
  {
    return determinant_device->evalGrad(P, iat);
  }
  ValueType ratioGrad(ParticleSet& P, int iat, GradType& grad)
  {
    return determinant_device->ratioGrad(P, iat, grad);
  }

  void evaluateGL(ParticleSet& P,
                  ParticleSet::ParticleGradient_t& G,
                  ParticleSet::ParticleLaplacian_t& L,
                  bool fromscratch = false)
  {
    determinant_device->evaluateGL(P, G, L, fromscratch);
  }

  inline void recompute()
  {
    determinant_device->recompute();
  }
  
  inline ValueType ratio(ParticleSet& P, int iel)
  {
    return determinant_device->ratio(P, iel);
  }
  
  inline void acceptMove(ParticleSet& P, int iel) {
    determinant_device->acceptMove(P, iel);
  }

  // accessor functions for checking
  inline double operator()(int i) {
    return determinant_device->operator()(i);
  }
  
  inline int size() const { determinant_device->size(); }

private:
  DeterminantDevice<DEVICE>* determinant_device;
};

} // namespace qmcplusplus

#endif
