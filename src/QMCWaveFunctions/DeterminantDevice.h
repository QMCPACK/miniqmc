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
// -*- C++ -*-

/**
 * @file
 * @brief CRTP base class for Determinant Devices
 */

#ifndef QMCPLUSPLUS_DETERMINANT_DEVICE_H
#define QMCPLUSPLUS_DETERMINANT_DEVICE_H

#include "clean_inlining.h"
//#include <impl/Kokkos_Timer.hpp>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#ifdef KOKKOS_ENABLE_CUDA
#include "cublas_v2.h"
#include "cusolverDn.h"
#endif
#include "Utilities/Configuration.h"
#include "QMCWaveFunctions/WaveFunctionComponent.h"
#include "Utilities/RandomGenerator.h"
#include "Memory/DeviceBuffers.hpp"
namespace qmcplusplus
{
template<class DEVICEIMP>
class DeterminantDevice
{
public:
  using QMCT = QMCTraits;

    DeterminantDevice(int nels, const RandomGenerator<QMCT::RealType>& RNG,
		    int First = 0) {}
  
  void checkMatrix()
  {
    static_cast<DEVICEIMP*>(this)->checkMatrixImp();
  }

  QMCT::RealType evaluateLog(ParticleSet& P,
		       ParticleSet::ParticleGradient_t& G,
		       ParticleSet::ParticleLaplacian_t& L)
  {
    return static_cast<DEVICEIMP*>(this)->evaluateLogImp(P, G, L);
  }

  QMCT::GradType evalGrad(ParticleSet& P, int iat)
  {
    return static_cast<DEVICEIMP*>(this)->evalGradImp(P, iat);
  }
  
  QMCT::ValueType ratioGrad(ParticleSet& P, int iat, QMCT::GradType& grad)
  {
    return static_cast<DEVICEIMP*>(this)->ratioImp(P, iat);
  }
  
  void evaluateGL(ParticleSet& P,
                  ParticleSet::ParticleGradient_t& G,
                  ParticleSet::ParticleLaplacian_t& L,
                  bool fromscratch = false)
  {
    static_cast<DEVICEIMP*>(this)->evaluateGLImp(P, G, L, fromscratch);
  }
  
  inline void recompute()
  {
    static_cast<DEVICEIMP*>(this)->recomputeImp();
  }
  
  inline QMCT::ValueType ratio(ParticleSet& P, int iel)
  {
    return static_cast<DEVICEIMP*>(this)->ratioImp(P, iel);
  }
  
  inline void acceptMove(ParticleSet& P, int iel) {
    static_cast<DEVICEIMP*>(this)->acceptMoveImp(P, iel);
  }

  /** accessor functions for checking?
   *
   *  would like to have const on here but with CRTP...
   */
  inline double operator()(int i) 
  {
    return static_cast<DEVICEIMP*>(this)->operatorParImp(i);
  }

    inline void finishUpdate(int iel)
    	{
    	     static_cast<DEVICEIMP*>(this)->finishUpdate_i(iel);
    	}
  
  inline int size() const { return static_cast<DEVICEIMP*>(this)->sizeImp(); }
};
} // namespace qmcplusplus

#endif
