//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by: Bryan Clark, bclark@Princeton.edu, Princeton University
//                    Ken Esler, kpesler@gmail.com, University of Illinois at Urbana-Champaign
//                    Miguel Morales, moralessilva2@llnl.gov, Lawrence Livermore National Laboratory
//                    Jeremy McMinnis, jmcminis@gmail.com, University of Illinois at Urbana-Champaign
//                    Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//                    Raymond Clay III, j.k.rofling@gmail.com, Lawrence Livermore National Laboratory
//                    Mark A. Berrill, berrillma@ornl.gov, Oak Ridge National Laboratory
//
// File created by: Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//////////////////////////////////////////////////////////////////////////////////////


/**@file DiracDeterminant.h
 * @brief Declaration of DiracDeterminant with a S(ingle)P(article)O(rbital)Set
 */
#ifndef QMCPLUSPLUS_DIRACDETERMINANT_H
#define QMCPLUSPLUS_DIRACDETERMINANT_H

#include "QMCWaveFunctions/WaveFunctionComponent.h"
#include "QMCWaveFunctions/SPOSet.h"
#include "Utilities/NewTimer.h"
#include "QMCWaveFunctions/DelayedUpdate.h"
#if defined(ENABLE_CUDA)
#include "QMCWaveFunctions/DelayedUpdateCUDA.h"
#endif

namespace qmcplusplus
{
template<typename DU_TYPE = DelayedUpdate<QMCTraits::ValueType, QMCTraits::QTFull::ValueType>>
class DiracDeterminant : public WaveFunctionComponent
{
public:
  using ValueVector_t = Vector<ValueType>;
  using ValueMatrix_t = Matrix<ValueType>;
  using GradVector_t  = Vector<GradType>;
  using GradMatrix_t  = Matrix<GradType>;

  using mValueType       = QMCTraits::QTFull::ValueType;
  using ValueMatrix_hp_t = Matrix<mValueType>;
  using mGradType        = TinyVector<mValueType, DIM>;

  /** constructor
   *@param spos the single-particle orbital set
   *@param first index of the first particle
   */
  DiracDeterminant(SPOSet* const spos, int first = 0, int delay = 1);

  // copy constructor and assign operator disabled
  DiracDeterminant(const DiracDeterminant& s) = delete;
  DiracDeterminant& operator=(const DiracDeterminant& s) = delete;

  ///invert psiM or its copies
  void invertPsiM(const ValueMatrix_t& logdetT, ValueMatrix_t& invMat);

  void evaluateGL(ParticleSet& P,
                  ParticleSet::ParticleGradient_t& G,
                  ParticleSet::ParticleLaplacian_t& L,
                  bool fromscratch = false);

  /** return the ratio only for the  iat-th partcle move
   * @param P current configuration
   * @param iat the particle thas is being moved
   */
  ValueType ratio(ParticleSet& P, int iat);

  /** compute multiple ratios for a particle move
  void evaluateRatios(VirtualParticleSet& VP, std::vector<ValueType>& ratios);
   */

  ValueType ratioGrad(ParticleSet& P, int iat, GradType& grad_iat);
  GradType evalGrad(ParticleSet& P, int iat);

  /** move was accepted, update the real container
   */
  void acceptMove(ParticleSet& P, int iat);
  void completeUpdates();

  ///evaluate log of a determinant for a particle set
  RealType evaluateLog(ParticleSet& P, ParticleSet::ParticleGradient_t& G, ParticleSet::ParticleLaplacian_t& L);

  void recompute(ParticleSet& P);

  /// psiM(j,i) \f$= \psi_j({\bf r}_i)\f$
  ValueMatrix_t psiM_temp;

  /// inverse transpose of psiM(j,i) \f$= \psi_j({\bf r}_i)\f$
  ValueMatrix_t psiM;

  /// dpsiM(i,j) \f$= \nabla_i \psi_j({\bf r}_i)\f$
  GradMatrix_t dpsiM;

  /// d2psiM(i,j) \f$= \nabla_i^2 \psi_j({\bf r}_i)\f$
  ValueMatrix_t d2psiM;

  /// value of single-particle orbital for particle-by-particle update
  ValueVector_t psiV;
  GradVector_t dpsiV;
  ValueVector_t d2psiV;

  /// delayed update engine
  DU_TYPE updateEng;

  /// the row of up-to-date inverse matrix
  ValueVector_t invRow;

  /** row id correspond to the up-to-date invRow. [0 norb), invRow is ready; -1, invRow is not valid.
   *  This id is set after calling getInvRow indicating invRow has been prepared for the invRow_id row
   *  ratioGrad checks if invRow_id is consistent. If not, invRow needs to be recomputed.
   *  acceptMove and completeUpdates mark invRow invalid by setting invRow_id to -1
   */
  int invRow_id;

  ValueType curRatio;

private:
  /// Timers
  NewTimer UpdateTimer, RatioTimer, InverseTimer, BufferTimer, SPOVTimer, SPOVGLTimer;
  /// a set of single-particle orbitals used to fill in the  values of the matrix
  SPOSet* const Phi;
  ///index of the first particle with respect to the particle set
  int FirstIndex;
  ///index of the last particle with respect to the particle set
  int LastIndex;
  ///number of single-particle orbitals which belong to this Dirac determinant
  int NumOrbitals;
  ///number of particles which belong to this Dirac determinant
  int NumPtcls;
  /// delayed update rank
  int ndelay;


  ///reset the size: with the number of particles and number of orbtials
  void resize(int nel, int morb);

  /// register all the timers
  void registerTimers()
  {
    UpdateTimer.reset();
    RatioTimer.reset();
    TimerManager.addTimer(&UpdateTimer);
    TimerManager.addTimer(&RatioTimer);
    TimerManager.addTimer(&InverseTimer);
    TimerManager.addTimer(&BufferTimer);
    TimerManager.addTimer(&SPOVTimer);
    TimerManager.addTimer(&SPOVGLTimer);
  }
};


} // namespace qmcplusplus
#endif
