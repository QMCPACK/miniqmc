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
//                    Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//                    Mark A. Berrill, berrillma@ornl.gov, Oak Ridge National Laboratory
//                    Jeongnim Kim, jeongnim.kim@intel.com, Intel Corp.
//
// File created by: Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//////////////////////////////////////////////////////////////////////////////////////


#include "QMCWaveFunctions/DiracDeterminant.h"
//#include "Numerics/DeterminantOperators.h"
#include "Numerics/OhmmsBlas.h"
//#include "Numerics/MatrixOperators.h"
//#include "simd/simd.hpp"

namespace qmcplusplus
{
/** constructor
 *@param spos the single-particle orbital set
 *@param first index of the first particle
 */
template<typename DU_TYPE>
DiracDeterminant<DU_TYPE>::DiracDeterminant(SPOSet* const spos, int first, int delay)
    : ndelay(1), invRow_id(-1),
        Phi(spos),
        FirstIndex(first),
        LastIndex(first + spos->size()),
        ndelay(delay),
        NumPtcls(spos->size()),
        NumOrbitals(spos->size()),
        UpdateTimer("DiracDeterminantBase::update", timer_level_fine),
        RatioTimer("DiracDeterminantBase::ratio", timer_level_fine),
        InverseTimer("DiracDeterminantBase::inverse", timer_level_fine),
        BufferTimer("DiracDeterminantBase::buffer", timer_level_fine),
        SPOVTimer("DiracDeterminantBase::spoval", timer_level_fine),
        SPOVGLTimer("DiracDeterminantBase::spovgl", timer_level_fine)
  {
    registerTimers();
    resize(spos->size(), spos->size());
  }

template<typename DU_TYPE>
void DiracDeterminant<DU_TYPE>::invertPsiM(const ValueMatrix_t& logdetT, ValueMatrix_t& invMat)
{
  InverseTimer.start();
  updateEng.invert_transpose(logdetT, invMat, LogValue, PhaseValue);
  InverseTimer.stop();
}


///reset the size: with the number of particles and number of orbtials
template<typename DU_TYPE>
void DiracDeterminant<DU_TYPE>::resize(int nel, int morb)
{
  int norb = morb;
  if (norb <= 0)
    norb = nel; // for morb == -1 (default)
  updateEng.resize(norb, ndelay);
  psiM.resize(nel, norb);
  dpsiM.resize(nel, norb);
  d2psiM.resize(nel, norb);
  psiV.resize(norb);
  invRow.resize(norb);
  psiM_temp.resize(nel, norb);
  LastIndex   = FirstIndex + nel;
  NumPtcls    = nel;
  NumOrbitals = norb;

  dpsiV.resize(NumOrbitals);
  d2psiV.resize(NumOrbitals);
}

template<typename DU_TYPE>
typename DiracDeterminant<DU_TYPE>::GradType DiracDeterminant<DU_TYPE>::evalGrad(ParticleSet& P, int iat)
{
  const int WorkingIndex = iat - FirstIndex;
  RatioTimer.start();
  invRow_id = WorkingIndex;
  updateEng.getInvRow(psiM, WorkingIndex, invRow);
  GradType g = simd::dot(invRow.data(), dpsiM[WorkingIndex], invRow.size());
  RatioTimer.stop();
  return g;
}

template<typename DU_TYPE>
typename DiracDeterminant<DU_TYPE>::ValueType DiracDeterminant<DU_TYPE>::ratioGrad(ParticleSet& P,
                                                                                   int iat,
                                                                                   GradType& grad_iat)
{
  SPOVGLTimer.start();
  Phi->evaluate(P, iat, psiV, dpsiV, d2psiV);
  SPOVGLTimer.stop();
  RatioTimer.start();
  const int WorkingIndex = iat - FirstIndex;
  UpdateMode             = ORB_PBYP_PARTIAL;
  GradType rv;

  // This is an optimization.
  // check invRow_id against WorkingIndex to see if getInvRow() has been called already
  // Some code paths call evalGrad before calling ratioGrad.
  if (invRow_id != WorkingIndex)
  {
    invRow_id = WorkingIndex;
    updateEng.getInvRow(psiM, WorkingIndex, invRow);
  }
  curRatio = simd::dot(invRow.data(), psiV.data(), invRow.size());
  grad_iat += ((RealType)1.0 / curRatio) * simd::dot(invRow.data(), dpsiV.data(), invRow.size());
  RatioTimer.stop();
  return curRatio;
}

/** move was accepted, update the real container
*/
template<typename DU_TYPE>
void DiracDeterminant<DU_TYPE>::acceptMove(ParticleSet& P, int iat)
{
  const int WorkingIndex = iat - FirstIndex;
  PhaseValue += evaluatePhase(curRatio);
  LogValue += std::log(std::abs(curRatio));
  UpdateTimer.start();
  updateEng.acceptRow(psiM, WorkingIndex, psiV);
  // invRow becomes invalid after accepting a move
  invRow_id = -1;
  if (UpdateMode == ORB_PBYP_PARTIAL)
  {
    simd::copy(dpsiM[WorkingIndex], dpsiV.data(), NumOrbitals);
    simd::copy(d2psiM[WorkingIndex], d2psiV.data(), NumOrbitals);
  }
  UpdateTimer.stop();
  curRatio = 1.0;
}

template<typename DU_TYPE>
void DiracDeterminant<DU_TYPE>::completeUpdates()
{
  UpdateTimer.start();
  // invRow becomes invalid after updating the inverse matrix
  invRow_id = -1;
  updateEng.updateInvMat(psiM);
  UpdateTimer.stop();
}

template<typename DU_TYPE>
void DiracDeterminant<DU_TYPE>::evaluateGL(ParticleSet& P,
                                           ParticleSet::ParticleGradient_t& G,
                                           ParticleSet::ParticleLaplacian_t& L,
                                           bool fromscratch)
{
  if (UpdateMode == ORB_PBYP_RATIO)
  { //need to compute dpsiM and d2psiM. Do not touch psiM!
    SPOVGLTimer.start();
    Phi->evaluate_notranspose(P, FirstIndex, LastIndex, psiM_temp, dpsiM, d2psiM);
    SPOVGLTimer.stop();
  }

  if (NumPtcls == 1)
  {
    ValueType y = psiM(0, 0);
    GradType rv = y * dpsiM(0, 0);
    G[FirstIndex] += rv;
    L[FirstIndex] += y * d2psiM(0, 0) - dot(rv, rv);
  }
  else
  {
    for (size_t i = 0, iat = FirstIndex; i < NumPtcls; ++i, ++iat)
    {
      mValueType dot_temp = simd::dot(psiM[i], d2psiM[i], NumOrbitals);
      mGradType rv        = simd::dot(psiM[i], dpsiM[i], NumOrbitals);
      G[iat] += rv;
      L[iat] += dot_temp - dot(rv, rv);
    }
  }
}

/** return the ratio only for the  iat-th partcle move
 * @param P current configuration
 * @param iat the particle thas is being moved
 */
template<typename DU_TYPE>
typename DiracDeterminant<DU_TYPE>::ValueType DiracDeterminant<DU_TYPE>::ratio(ParticleSet& P, int iat)
{
  UpdateMode             = ORB_PBYP_RATIO;
  const int WorkingIndex = iat - FirstIndex;
  SPOVTimer.start();
  Phi->evaluate(P, iat, psiV);
  SPOVTimer.stop();
  RatioTimer.start();
  // This is an optimization.
  // check invRow_id against WorkingIndex to see if getInvRow() has been called
  // This is intended to save redundant compuation in TM1 and TM3
  if (invRow_id != WorkingIndex)
  {
    invRow_id = WorkingIndex;
    updateEng.getInvRow(psiM, WorkingIndex, invRow);
  }
  curRatio = simd::dot(invRow.data(), psiV.data(), invRow.size());
  RatioTimer.stop();
  return curRatio;
}

/* Ye: to be recovered
template<typename DU_TYPE>
void DiracDeterminant<DU_TYPE>::evaluateRatios(VirtualParticleSet& VP, std::vector<ValueType>& ratios)
{
  SPOVTimer.start();
  const int WorkingIndex = VP.refPtcl - FirstIndex;
  invRow_id              = WorkingIndex;
  updateEng.getInvRow(psiM, WorkingIndex, invRow);
  Phi->evaluateDetRatios(VP, psiV, invRow, ratios);
  SPOVTimer.stop();
}
*/

/** Calculate the log value of the Dirac determinant for particles
 *@param P input configuration containing N particles
 *@param G a vector containing N gradients
 *@param L a vector containing N laplacians
 *@return the value of the determinant
 *
 *\f$ (first,first+nel). \f$  Add the gradient and laplacian
 *contribution of the determinant to G(radient) and L(aplacian)
 *for local energy calculations.
 */
template<typename DU_TYPE>
typename DiracDeterminant<DU_TYPE>::RealType DiracDeterminant<DU_TYPE>::evaluateLog(ParticleSet& P,
                                                                                    ParticleSet::ParticleGradient_t& G,
                                                                                    ParticleSet::ParticleLaplacian_t& L)
{
  recompute(P);

  if (NumPtcls == 1)
  {
    ValueType y = psiM(0, 0);
    GradType rv = y * dpsiM(0, 0);
    G[FirstIndex] += rv;
    L[FirstIndex] += y * d2psiM(0, 0) - dot(rv, rv);
  }
  else
  {
    for (int i = 0, iat = FirstIndex; i < NumPtcls; i++, iat++)
    {
      mGradType rv   = simd::dot(psiM[i], dpsiM[i], NumOrbitals);
      mValueType lap = simd::dot(psiM[i], d2psiM[i], NumOrbitals);
      G[iat] += rv;
      L[iat] += lap - dot(rv, rv);
    }
  }
  return LogValue;
}

template<typename DU_TYPE>
void DiracDeterminant<DU_TYPE>::recompute(ParticleSet& P)
{
  SPOVGLTimer.start();
  Phi->evaluate_notranspose(P, FirstIndex, LastIndex, psiM_temp, dpsiM, d2psiM);
  SPOVGLTimer.stop();
  if (NumPtcls == 1)
  {
    //CurrentDet=psiM(0,0);
    ValueType det = psiM_temp(0, 0);
    psiM(0, 0)    = RealType(1) / det;
    LogValue      = evaluateLogAndPhase(det, PhaseValue);
  }
  else
  {
    invertPsiM(psiM_temp, psiM);
  }
}

typedef QMCTraits::ValueType ValueType;
typedef QMCTraits::QTFull::ValueType mValueType;

template class DiracDeterminant<>;
#if defined(ENABLE_CUDA)
template class DiracDeterminant<DelayedUpdateCUDA<ValueType, mValueType>>;
#endif

} // namespace qmcplusplus
