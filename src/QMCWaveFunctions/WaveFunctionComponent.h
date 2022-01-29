////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by:
// Ken Esler, kpesler@gmail.com,
//    University of Illinois at Urbana-Champaign
// Miguel Morales, moralessilva2@llnl.gov,
//    Lawrence Livermore National Laboratory
// Jeremy McMinnis, jmcminis@gmail.com,
//    University of Illinois at Urbana-Champaign
// Jeongnim Kim, jeongnim.kim@gmail.com,
//    University of Illinois at Urbana-Champaign
// Raymond Clay III, j.k.rofling@gmail.com,
//    Lawrence Livermore National Laboratory
// Mark A. Berrill, berrillma@ornl.gov,
//    Oak Ridge National Laboratory
//
// File created by:
// Jeongnim Kim, jeongnim.kim@gmail.com,
//    University of Illinois at Urbana-Champaign
////////////////////////////////////////////////////////////////////////////////

#ifndef QMCPLUSPLUS_WAVEFUNCTIONCOMPONENTBASE_H
#define QMCPLUSPLUS_WAVEFUNCTIONCOMPONENTBASE_H
#include "Utilities/Configuration.h"
#include "Particle/ParticleSet.h"
#include "Particle/VirtualParticleSet.h"
#include "Particle/DistanceTableData.h"

/**@file WaveFunctionComponent.h
 *@brief Declaration of WaveFunctionComponent
 */
namespace qmcplusplus
{
/// forward declaration of WaveFunctionComponent
struct WaveFunctionComponent;

typedef WaveFunctionComponent* WaveFunctionComponentPtr;

/**@defgroup WaveFunctionComponent Wavefunction Component group
 * @brief Classes which constitute a many-body trial wave function
 *
 * A many-body trial wave function is
 * \f[
 \Psi(\{ {\bf R}\}) = \prod_i \psi_{i}(\{ {\bf R}\}),
 * \f]
 * where \f$\Psi\f$s are represented by
 * the derived classes from WaveFunctionComponent.
 */
/** @ingroup WaveFunctionComponentComponent
 * @brief An abstract class for a component of a many-body trial wave function
 */
struct WaveFunctionComponent : public QMCTraits
{
  /** enum for a update mode */
  enum
  {
    ORB_PBYP_RATIO,   /*!< particle-by-particle ratio only */
    ORB_PBYP_ALL,     /*!< particle-by-particle, update Value-Gradient-Laplacian */
    ORB_PBYP_PARTIAL, /*!< particle-by-particle, update Value and Grdient */
    ORB_WALKER,       /*!< walker update */
    ORB_ALLWALKER     /*!< all walkers update */
  };

  typedef ParticleAttrib<ValueType> ValueVectorType;
  typedef ParticleAttrib<GradType> GradVectorType;
  typedef PooledData<RealType> BufferType;
  typedef ParticleSet::Walker_t Walker_t;

  /** flag to set the optimization mode */
  bool IsOptimizing;
  /** boolean to set optimization
   *
   * If true, this object is actively modified during optimization
   */
  bool Optimizable;
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
  /** Name of this wavefunction component
   */
  std::string WaveFunctionComponentName;

  /// default constructor
  WaveFunctionComponent()
      : IsOptimizing(false),
        Optimizable(true),
        UpdateMode(ORB_WALKER),
        LogValue(0.0),
        PhaseValue(0.0),
        WaveFunctionComponentName("WaveFunctionComponent")
  {}

  /// default destructor
  virtual ~WaveFunctionComponent() {}

  /// operates on a single walker

  /** evaluate the value of the wavefunction
   * @param P active ParticleSet
   * @param G Gradients, \f$\nabla\ln\Psi\f$
   * @param L Laplacians, \f$\nabla^2\ln\Psi\f$
   *
   */
  virtual RealType evaluateLog(ParticleSet& P,
                               ParticleSet::ParticleGradient_t& G,
                               ParticleSet::ParticleLaplacian_t& L) = 0;

  /** return the current gradient for the iat-th particle
   * @param Pquantum particle set
   * @param iat particle index
   * @return the gradient of the iat-th particle
   */
  virtual GradType evalGrad(ParticleSet& P, int iat) = 0;

  /** evaluate the ratio of the new to old wavefunction component value
   * @param P the active ParticleSet
   * @param iat the index of a particle
   * @param grad_iat Gradient for the active particle
   */
  virtual ValueType ratioGrad(ParticleSet& P, int iat, GradType& grad_iat) = 0;

  /** a move for iat-th particle is accepted. Update the content for the next
   * moves
   * @param P target ParticleSet
   * @param iat index of the particle whose new position was proposed
   */
  virtual void acceptMove(ParticleSet& P, int iat) = 0;

  /** evalaute the ratio of the new to old wavefunction component value
   *@param P the active ParticleSet
   *@param iat the index of a particle
   *@return \f$ \psi( \{ {\bf R}^{'} \} )/ \psi( \{ {\bf R}^{'}\})\f$
   *
   *Specialized for particle-by-particle move.
   */
  virtual ValueType ratio(ParticleSet& P, int iat) = 0;

  /** compute G and L after the sweep
   * @param P active ParticleSet
   * @param G Gradients, \f$\nabla\ln\Psi\f$
   * @param L Laplacians, \f$\nabla^2\ln\Psi\f$
   * @param fromscratch, recompute internal data if true
   *
   */
  virtual void evaluateGL(ParticleSet& P,
                          ParticleSet::ParticleGradient_t& G,
                          ParticleSet::ParticleLaplacian_t& L,
                          bool fromscratch = false) = 0;

  /** complete all the delayed updates, must be called after each substep or step during pbyp move
   */
  virtual void completeUpdates(){};

  /** evaluate ratios to evaluate the non-local PP
   * @param VP VirtualParticleSet
   * @param ratios ratios with new positions VP.R[k] the VP.refPtcl
   */
  virtual void evaluateRatios(VirtualParticleSet& VP, std::vector<ValueType>& ratios) = 0;

  /// operates on multiple walkers
  virtual void multi_evaluateLog(const std::vector<WaveFunctionComponent*>& WFC_list,
                                 const std::vector<ParticleSet*>& P_list,
                                 const std::vector<ParticleSet::ParticleGradient_t*>& G_list,
                                 const std::vector<ParticleSet::ParticleLaplacian_t*>& L_list,
                                 ParticleSet::ParticleValue_t& values)
  {
#pragma omp parallel for
    for (int iw = 0; iw < P_list.size(); iw++)
      values[iw] = WFC_list[iw]->evaluateLog(*P_list[iw], *G_list[iw], *L_list[iw]);
  };

  virtual void multi_evalGrad(const std::vector<WaveFunctionComponent*>& WFC_list,
                              const std::vector<ParticleSet*>& P_list,
                              int iat,
                              std::vector<PosType>& grad_now)
  {
    //#pragma omp parallel for
    for (int iw = 0; iw < P_list.size(); iw++)
      grad_now[iw] = WFC_list[iw]->evalGrad(*P_list[iw], iat);
  };

  virtual void multi_ratioGrad(const std::vector<WaveFunctionComponent*>& WFC_list,
                               const std::vector<ParticleSet*>& P_list,
                               int iat,
                               std::vector<ValueType>& ratios,
                               std::vector<PosType>& grad_new)
  {
#pragma omp parallel for
    for (int iw = 0; iw < P_list.size(); iw++)
      ratios[iw] = WFC_list[iw]->ratioGrad(*P_list[iw], iat, grad_new[iw]);
  };

  virtual void multi_acceptrestoreMove(const std::vector<WaveFunctionComponent*>& WFC_list,
                                       const std::vector<ParticleSet*>& P_list,
                                       const std::vector<bool>& isAccepted,
                                       int iat)
  {
#pragma omp parallel for
    for (int iw = 0; iw < P_list.size(); iw++)
    {
      if (isAccepted[iw])
        WFC_list[iw]->acceptMove(*P_list[iw], iat);
    }
  };

  virtual void multi_ratio(const std::vector<WaveFunctionComponent*>& WFC_list,
                           const std::vector<ParticleSet*>& P_list,
                           int iat,
                           ParticleSet::ParticleValue_t& ratio_list){
      // TODO
  };

  virtual void multi_evaluateGL(const std::vector<WaveFunctionComponent*>& WFC_list,
                                const std::vector<ParticleSet*>& P_list,
                                const std::vector<ParticleSet::ParticleGradient_t*>& G_list,
                                const std::vector<ParticleSet::ParticleLaplacian_t*>& L_list,
                                bool fromscratch = false)
  {
#pragma omp parallel for
    for (int iw = 0; iw < P_list.size(); iw++)
      WFC_list[iw]->evaluateGL(*P_list[iw], *G_list[iw], *L_list[iw], fromscratch);
  };

  virtual void multi_completeUpdates(const std::vector<WaveFunctionComponent*>& WFC_list)
  {
#pragma omp parallel for
    for (int iw = 0; iw < WFC_list.size(); iw++)
      WFC_list[iw]->completeUpdates();
  }
};
} // namespace qmcplusplus
#endif
