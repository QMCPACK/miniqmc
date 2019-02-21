////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2019 QMCPACK developers.
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
 * @brief CPU implementation of Determinant
 */

#ifndef QMCPLUSPLUS_DETERMINANT_DEVICE_IMP_CUDA_H
#define QMCPLUSPLUS_DETERMINANT_DEVICE_IMP_CUDA_H

#include "Devices.h"
#include "Numerics/OhmmsPETE/OhmmsMatrix.h"
#include "Numerics/DeterminantOperators.h"
#include "DeterminantDevice.h"
#include "DeterminantDeviceImp.h"
#include "QMCWaveFunctions/WaveFunctionComponent.h"
#include "Numerics/LinAlgCPU.h"
#include "Utilities/Configuration.h"

namespace qmcplusplus
{


template<>
class DeterminantDeviceImp<Devices::CUDA>
  : public DeterminantDevice<DeterminantDeviceImp<Devices::CUDA>>,
    public LinAlgCPU
{
public:
  using QMCT = QMCTraits;
  
  DeterminantDeviceImp(int nels, const RandomGenerator<QMCT::RealType>& RNG, int First = 0)
    : DeterminantDevice( nels, RNG, First),
      FirstIndex(First),
      myRandom(RNG)
  {
    psiMinv.resize(nels, nels);
    psiV.resize(nels);
    psiM.resize(nels, nels);

    pivot.resize(nels);
    psiMsave.resize(nels, nels);

    // now we "void initialize(RandomGenerator<T> RNG)"

    nels = psiM.rows();
    // get lwork and resize workspace
    LWork = getGetriWorkspace(psiM.data(), nels, nels, pivot.data());
    work.resize(LWork);

    constexpr double shift(0.5);
    myRandom.generate_uniform(psiMsave.data(), nels * nels);
    psiMsave -= shift;

    double phase;
    transpose(psiMsave.data(), psiM.data(), nels, nels);
    LogValue = InvertWithLog(psiM.data(), nels, nels, work.data(), LWork, pivot.data(), phase);
    std::copy_n(psiM.data(), nels * nels, psiMinv.data());
  }

  void checkMatrixImp()
  {
    if (omp_get_num_threads() == 1)
    {
      checkIdentity(psiMsave, psiM, "Psi_0 * psiM(double)");
      checkIdentity(psiMsave, psiMinv, "Psi_0 * psiMinv(T)");
      checkDiffCPU(psiM, psiMinv, "psiM(double)-psiMinv(T)");
    }
  }

  QMCT::RealType evaluateLogImp(ParticleSet& P,
                       ParticleSet::ParticleGradient_t& G,
                       ParticleSet::ParticleLaplacian_t& L)
  {
    recompute();
    // FIXME do we want remainder of evaluateLog?
    return 0.0;
  }

  QMCT::GradType evalGradImp(ParticleSet& P, int iat) { return QMCT::GradType(); }

  QMCT::ValueType ratioGradImp(ParticleSet& P, int iat, QMCT::GradType& grad) { return ratio(P, iat); }

  void evaluateGLImp(ParticleSet& P,
                  ParticleSet::ParticleGradient_t& G,
                  ParticleSet::ParticleLaplacian_t& L,
                  bool fromscratch = false)
  {}

  /// recompute the inverse
  inline void recomputeImp()
  {
    const int nels = psiV.size();
    transpose(psiMsave.data(), psiM.data(), nels, nels);
    InvertOnly(psiM.data(), nels, nels, work.data(), pivot.data(), LWork);
    std::copy_n(psiM.data(), nels * nels, psiMinv.data());
  }

  /** return determinant ratio for the row replacement
   * @param iel the row (active particle) index
   */
  inline QMCT::ValueType ratioImp(ParticleSet& P, int iel)
  {
    const int nels = psiV.size();
    constexpr double shift(0.5);
    constexpr double czero(0);
    for (int j = 0; j < nels; ++j)
      psiV[j] = myRandom() - shift;
    curRatio = inner_product_n(psiV.data(), psiMinv[iel - FirstIndex], nels, czero);
    return curRatio;
  }

  /** accept the row and update the inverse */
  inline void acceptMoveImp(ParticleSet& P, int iel)
  {
    const int nels = psiV.size();
    updateRow(psiMinv.data(), psiV.data(), nels, nels, iel - FirstIndex, curRatio);
    std::copy_n(psiV.data(), nels, psiMsave[iel - FirstIndex]);
  }

  /** accessor functions for checking */
  inline double operatorParImp(int i) const { return psiMinv(i); }
  inline int sizeImp() const { return psiMinv.size(); }

private:
  /// log|det|
  double LogValue;
  /// current ratio
  double curRatio;
  /// workspace size
  int LWork;
  /// initial particle index
  const int FirstIndex;
  /// inverse matrix to be update
  Matrix<QMCT::RealType> psiMinv;
  /// a SPO set for the row update
  aligned_vector<QMCT::RealType> psiV;
  /// internal storage to perform inversion correctly
  Matrix<double> psiM; // matrix to be inverted
  /// random number generator for testing
  RandomGenerator<QMCT::RealType> myRandom;

  // temporary workspace for inversion
  aligned_vector<int> pivot;
  aligned_vector<double> work;
  Matrix<QMCT::RealType> psiMsave;
};

extern template class DeterminantDeviceImp<Devices::CUDA>;

} // namespace qmcplusplus

#endif
