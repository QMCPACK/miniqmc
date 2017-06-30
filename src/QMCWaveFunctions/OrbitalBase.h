//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by: Ken Esler, kpesler@gmail.com, University of Illinois at Urbana-Champaign
//                    Miguel Morales, moralessilva2@llnl.gov, Lawrence Livermore National Laboratory
//                    Jeremy McMinnis, jmcminis@gmail.com, University of Illinois at Urbana-Champaign
//                    Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//                    Raymond Clay III, j.k.rofling@gmail.com, Lawrence Livermore National Laboratory
//                    Mark A. Berrill, berrillma@ornl.gov, Oak Ridge National Laboratory
//
// File created by: Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//////////////////////////////////////////////////////////////////////////////////////
    
    
#ifndef QMCPLUSPLUS_ORBITALBASE_H
#define QMCPLUSPLUS_ORBITALBASE_H
#include "Configuration.h"
#include "Particle/ParticleSet.h"
#include "Particle/DistanceTableData.h"
#include "QMCWaveFunctions/OrbitalSetTraits.h"
#include "Particle/MCWalkerConfiguration.h"

/**@file OrbitalBase.h
 *@brief Declaration of OrbitalBase
 */
namespace qmcplusplus
{

///forward declaration of OrbitalBase
class OrbitalBase;

typedef OrbitalBase*                       OrbitalBasePtr;

/**@defgroup OrbitalComponent Orbital group
 * @brief Classes which constitute a many-body trial wave function
 *
 * A many-body trial wave function is
 * \f[
 \Psi(\{ {\bf R}\}) = \prod_i \psi_{i}(\{ {\bf R}\}),
 * \f]
 * where \f$\Psi\f$s are represented by
 * the derived classes from OrbtialBase.
 */
/** @ingroup OrbitalComponent
 * @brief An abstract class for a component of a many-body trial wave function
 */
struct OrbitalBase: public QMCTraits
{

  ///recasting enum of DistanceTableData to maintain consistency
  enum {SourceIndex  = DistanceTableData::SourceIndex,
        VisitorIndex = DistanceTableData::VisitorIndex,
        WalkerIndex  = DistanceTableData::WalkerIndex
       };

  /** enum for a update mode */
  enum
  {
    ORB_PBYP_RATIO,   /*!< particle-by-particle ratio only */
    ORB_PBYP_ALL,     /*!< particle-by-particle, update Value-Gradient-Laplacian */
    ORB_PBYP_PARTIAL, /*!< particle-by-particle, update Value and Grdient */
    ORB_WALKER,    /*!< walker update */
    ORB_ALLWALKER  /*!< all walkers update */
  };

  typedef ParticleAttrib<ValueType> ValueVectorType;
  typedef ParticleAttrib<GradType>  GradVectorType;
  typedef PooledData<RealType>      BufferType;
  typedef ParticleSet::Walker_t     Walker_t;
  typedef OrbitalSetTraits<RealType>::ValueMatrix_t       RealMatrix_t;
  typedef OrbitalSetTraits<ValueType>::ValueMatrix_t      ValueMatrix_t;
  typedef OrbitalSetTraits<ValueType>::GradMatrix_t       GradMatrix_t;
  typedef OrbitalSetTraits<ValueType>::HessType           HessType;
  typedef OrbitalSetTraits<ValueType>::HessVector_t       HessVector_t;

  /** flag to set the optimization mode */
  bool IsOptimizing;
  /** boolean to set optimization
   *
   * If true, this object is actively modified during optimization
   */
  bool Optimizable;
  /** true, if FermionWF */
  bool IsFermionWF;
  /** true, if compute for the ratio instead of buffering */
  bool Need2Compute4PbyP;

  /** current update mode */
  int UpdateMode;
  /** current \f$\log\phi \f$
   */
  RealType LogValue;
  /** current phase
   */
  RealType PhaseValue;
  /** A vector for \f$ \frac{\partial \nabla \log\phi}{\partial \alpha} \f$
   */
  GradVectorType dLogPsi;
  /** A vector for \f$ \frac{\partial \nabla^2 \log\phi}{\partial \alpha} \f$
   */
  ValueVectorType d2LogPsi;
  /** Name of this orbital
   */
  std::string OrbitalName;

  /// default constructor
  OrbitalBase():
  IsOptimizing(false), Optimizable(true), UpdateMode(ORB_WALKER), 
  LogValue(1.0), PhaseValue(0.0), OrbitalName("OrbitalBase")
  {
    ///store instead of computing
    Need2Compute4PbyP=false;
  }

  ///default destructor
  virtual ~OrbitalBase() { }

  /** evaluate the value of the orbital
   * @param P active ParticleSet
   * @param G Gradients, \f$\nabla\ln\Psi\f$
   * @param L Laplacians, \f$\nabla^2\ln\Psi\f$
   *
   */
  virtual RealType
  evaluateLog(ParticleSet& P,
              ParticleSet::ParticleGradient_t& G, ParticleSet::ParticleLaplacian_t& L) = 0;

  /** return the current gradient for the iat-th particle
   * @param Pquantum particle set
   * @param iat particle index
   * @return the gradient of the iat-th particle
   */
  virtual GradType evalGrad(ParticleSet& P, int iat) = 0;

  /** evaluate the ratio of the new to old orbital value
   * @param P the active ParticleSet
   * @param iat the index of a particle
   * @param grad_iat Gradient for the active particle
   */
  virtual ValueType ratioGrad(ParticleSet& P, int iat, GradType& grad_iat) = 0;

  /** a move for iat-th particle is accepted. Update the content for the next moves
   * @param P target ParticleSet
   * @param iat index of the particle whose new position was proposed
   */
  virtual void acceptMove(ParticleSet& P, int iat) =0;

  /** evalaute the ratio of the new to old orbital value
   *@param P the active ParticleSet
   *@param iat the index of a particle
   *@return \f$ \psi( \{ {\bf R}^{'} \} )/ \psi( \{ {\bf R}^{'}\})\f$
   *
   *Specialized for particle-by-particle move.
   */
  virtual ValueType ratio(ParticleSet& P, int iat) =0;

};
}
#endif

