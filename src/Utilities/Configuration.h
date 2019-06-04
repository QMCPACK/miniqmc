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
#include <Utilities/QMCTypes.h>
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
inline omp_int_t omp_get_level() { return 0; }
inline omp_int_t omp_get_ancestor_thread_num(int level) { return 0; }
inline bool omp_get_nested() { return false; }
#endif

/// get the number of threads at the next nested level
inline int getNumThreadsNested()
{
  int num_threads = 1;
  if (omp_get_nested())
  {
#pragma omp parallel
    {
#pragma omp master
      num_threads = omp_get_num_threads();
    }
  }
  return num_threads;
}

// define empty DEBUG_MEMORY
#define DEBUG_MEMORY(msg)
// uncomment this out to trace the call tree of destructors
//#define DEBUG_MEMORY(msg) std::cerr << "<<<< " << msg << std::endl;

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
  enum
  {
    DIM = OHMMS_DIM
  };
  using QTBase = QMCTypes<OHMMS_PRECISION, DIM>;
  using QTFull = QMCTypes<OHMMS_PRECISION_FULL, DIM>;
  typedef QTBase::RealType RealType;
  typedef QTBase::ComplexType ComplexType;
  typedef QTBase::ValueType ValueType;
  typedef QTBase::PosType PosType;
  typedef QTBase::GradType GradType;
  typedef QTBase::TensorType TensorType;
  ///define other types
  typedef OHMMS_INDEXTYPE IndexType;
  typedef QTFull::RealType EstimatorRealType;
};


/** Particle traits to use UniformGridLayout for the ParticleLayout.
 */
struct PtclOnLatticeTraits
{
  // clang-format off
  typedef CrystalLattice<OHMMS_PRECISION,3,OHMMS_ORTHO>  ParticleLayout_t;

  using QTFull = QMCTraits::QTFull;

  typedef int Index_t;
  typedef QTFull::RealType Scalar_t;
  typedef QTFull::ComplexType Complex_t;

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

// For unit tests
//  Check if we are compiling with Catch defined.  Could use other symbols if needed.
#ifdef TEST_CASE
#ifdef QMC_COMPLEX
typedef ComplexApprox ValueApprox;
#else
typedef Approx ValueApprox;
#endif
#endif

} // namespace qmcplusplus

#endif
