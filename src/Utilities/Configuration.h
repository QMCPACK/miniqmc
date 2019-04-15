////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by:
// Ken Esler, kpesler@gmail.com,
//    University of Illinois at Urbana-Champaign
// Jeongnim Kim, jeongnim.kim@gmail.com,
//    University of Illinois at Urbana-Champaign
// Jeremy McMinnis, jmcminis@gmail.com,
//    University of Illinois at Urbana-Champaign
//
// File created by:
// Jeongnim Kim, jeongnim.kim@gmail.com,
//    University of Illinois at Urbana-Champaign
////////////////////////////////////////////////////////////////////////////////

#ifndef QMCPLUSPLUS_TRAITS_H
#define QMCPLUSPLUS_TRAITS_H

#include <config.h>
#include <string>
#include <vector>
#include <map>
#include <complex>
#include <Numerics/OhmmsPETE/TinyVector.h>
#include <Numerics/OhmmsPETE/Tensor.h>
#include "Particle/Lattice/CrystalLattice.h"
#include <Particle/ParticleAttrib.h>
#include <Utilities/OutputManager.h>

#define APP_ABORT(msg)                                            \
  {                                                               \
    std::cerr << "Fatal Error. Aborting at " << msg << std::endl; \
    exit(1);                                                      \
  }

#if defined(ENABLE_OPENMP)
#include <omp.h>
#else
typedef int omp_int_t;
inline omp_int_t omp_get_thread_num() { return 0; }
inline omp_int_t omp_get_max_threads() { return 1; }
inline omp_int_t omp_get_num_threads() { return 1; }
inline omp_int_t omp_get_active_level() { return 0; }
inline omp_int_t omp_get_ancestor_thread_num(int level) { return 0; }
#endif

// define empty DEBUG_MEMORY
#define DEBUG_MEMORY(msg)
// uncomment this out to trace the call tree of destructors
//#define DEBUG_MEMORY(msg) std::cerr << "<<<< " << msg << std::endl;

#if defined(DEBUG_PSIBUFFER_ON)
#define DEBUG_PSIBUFFER(who, msg)                              \
  std::cerr << "PSIBUFFER " << who << " " << msg << std::endl; \
  std::cerr.flush();
#else
#define DEBUG_PSIBUFFER(who, msg)
#endif

namespace qmcplusplus
{
/** traits for the common particle attributes
 *
 *This is an alternative to the global typedefs.
 */
struct PtclAttribTraits
{
  // clang-format off
  typedef int                                                     Index_t;
  typedef ParticleAttrib<Index_t>                                 ParticleIndex_t;
  typedef ParticleAttrib<OHMMS_PRECISION>                         ParticleScalar_t;
  typedef ParticleAttrib<TinyVector<OHMMS_PRECISION, OHMMS_DIM> > ParticlePos_t;
  typedef ParticleAttrib<Tensor<OHMMS_PRECISION, OHMMS_DIM> >     ParticleTensor_t;
  // clang-format on
};

/** traits for QMC variables
 *
 *typedefs for the QMC data types
 */
struct QMCTraits
{
  // clang-format off
  enum {DIM = OHMMS_DIM};
  typedef OHMMS_INDEXTYPE                IndexType;
  typedef OHMMS_PRECISION                RealType;
  typedef OHMMS_PRECISION_FULL           EstimatorRealType;
#if defined(QMC_COMPLEX)
  typedef std::complex<OHMMS_PRECISION>  ValueType;
#else
  typedef OHMMS_PRECISION                ValueType;
#endif
  typedef std::complex<RealType>         ComplexType;
  typedef TinyVector<RealType,DIM>       PosType;
  typedef TinyVector<ValueType,DIM>      GradType;
  typedef Tensor<RealType,DIM>           TensorType;
  // clang-format on
};

/** Particle traits to use UniformGridLayout for the ParticleLayout.
 */
struct PtclOnLatticeTraits
{
  // clang-format off
  typedef CrystalLattice<OHMMS_PRECISION,3,OHMMS_ORTHO>  ParticleLayout_t;

  typedef int                                          Index_t;
  typedef OHMMS_PRECISION_FULL                         Scalar_t;
  typedef std::complex<Scalar_t>                       Complex_t;

  typedef ParticleLayout_t::SingleParticleIndex_t      SingleParticleIndex_t;
  typedef ParticleLayout_t::SingleParticlePos_t        SingleParticlePos_t;
  typedef ParticleLayout_t::Tensor_t                   Tensor_t;

  typedef ParticleAttrib<Index_t>                      ParticleIndex_t;
  typedef ParticleAttrib<Scalar_t>                     ParticleScalar_t;
  typedef ParticleAttrib<SingleParticlePos_t>          ParticlePos_t;
  typedef ParticleAttrib<Tensor_t>                     ParticleTensor_t;

#if defined(QMC_COMPLEX)
  typedef ParticleAttrib<TinyVector<Complex_t,OHMMS_DIM> > ParticleGradient_t;
  typedef ParticleAttrib<Complex_t>                      ParticleLaplacian_t;
  typedef ParticleAttrib<Complex_t>                      ParticleValue_t;
  typedef Complex_t                                      SingleParticleValue_t;
#else
  typedef ParticleAttrib<TinyVector<Scalar_t,OHMMS_DIM> > ParticleGradient_t;
  typedef ParticleAttrib<Scalar_t>                       ParticleLaplacian_t;
  typedef ParticleAttrib<Scalar_t>                       ParticleValue_t;
  typedef Scalar_t                                       SingleParticleValue_t;
#endif
  // clang-format on
};


} // namespace qmcplusplus

#endif
