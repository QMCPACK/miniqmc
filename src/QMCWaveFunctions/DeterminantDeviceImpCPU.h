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

#ifndef QMCPLUSPLUS_DETERMINANT_DEVICE_IMP_CPU_H
#define QMCPLUSPLUS_DETERMINANT_DEVICE_IMP_CPU_H
#include "Numerics/OhmmsPETE/OhmmsMatrix.h"
#include "Numerics/DeterminantOperators.h"
#include "QMCWaveFunctions/WaveFunctionComponent.h"
#include "Utilities/Configuration.h"

namespace qmcplusplus
{
namespace future
{
/**@{Determinant utilities */
/** Inversion of a double matrix after LU factorization*/
inline void
    getri(int n, double* restrict a, int lda, int* restrict piv, double* restrict work, int& lwork)
{
  int status;
  dgetri(n, a, lda, piv, work, lwork, status);
}

/** Inversion of a float matrix after LU factorization*/
inline void
    getri(int n, float* restrict a, int lda, int* restrict piv, float* restrict work, int& lwork)
{
  int status;
  sgetri(n, a, lda, piv, work, lwork, status);
}

/** Inversion of a std::complex<double> matrix after LU factorization*/
inline void getri(int n,
                  std::complex<double>* restrict a,
                  int lda,
                  int* restrict piv,
                  std::complex<double>* restrict work,
                  int& lwork)
{
  int status;
  zgetri(n, a, lda, piv, work, lwork, status);
}

/** Inversion of a complex<float> matrix after LU factorization*/
inline void getri(int n,
                  std::complex<float>* restrict a,
                  int lda,
                  int* restrict piv,
                  std::complex<float>* restrict work,
                  int& lwork)
{
  int status;
  cgetri(n, a, lda, piv, work, lwork, status);
}


/** query the size of workspace for Xgetri after LU decompisiton */
template<class T>
inline int getGetriWorkspace(T* restrict x, int n, int lda, int* restrict pivot)
{
  T work;
  int lwork = -1;
  getri(n, x, lda, pivot, &work, lwork);
  lwork = static_cast<int>(work);
  return lwork;
}

/** transpose in to out
 *
 * Assume: in[n][lda] and out[n][lda]
 */
template<typename TIN, typename TOUT>
inline void transpose(const TIN* restrict in, TOUT* restrict out, int n, int lda)
{
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      out[i * lda + j] = in[i + j * lda];
}

/// used only for debugging or walker move
template<class T>
inline T
    InvertWithLog(T* restrict x, int n, int lda, T* restrict work, int lwork, int* restrict pivot, T& phase)
{
  T logdet(0.0);
  LUFactorization(n, n, x, lda, pivot);
  int sign_det = 1;
  for (int i = 0; i < n; i++)
  {
    sign_det *= (pivot[i] == i + 1) ? 1 : -1;
    sign_det *= (x[i * lda + i] > 0) ? 1 : -1;
    logdet += std::log(std::abs(x[i * lda + i]));
  }
  getri(n, x, lda, pivot, work, lwork);
  phase = (sign_det > 0) ? 0.0 : M_PI;
  return logdet;
}

/// inner product
template<typename T1, typename T2, typename T3>
inline T3 inner_product_n(const T1* restrict a, const T2* restrict b, int n, T3 res)
{
  for (int i = 0; i < n; ++i)
    res += a[i] * b[i];
  return res;
}

/// recompute inverse, do not evaluate log|det|
template<class T>
inline void
    InvertOnly(T* restrict x, int n, int lda, T* restrict work, int* restrict pivot, int lwork)
{
  LUFactorization(n, n, x, lda, pivot);
  getri(n, x, lda, pivot, work, lwork);
}

/** update Row as implemented in the full code */
/** [UpdateRow] */
template<typename T, typename RT>
inline void
    updateRow(T* restrict pinv, const T* restrict tv, int m, int lda, int rowchanged, RT c_ratio_in)
{
  constexpr T cone(1);
  constexpr T czero(0);
  T temp[m], rcopy[m];
  T c_ratio = cone / c_ratio_in;
  BLAS::gemv('T', m, m, c_ratio, pinv, m, tv, 1, czero, temp, 1);
  temp[rowchanged] = cone - c_ratio;
  std::copy_n(pinv + m * rowchanged, m, rcopy);
  BLAS::ger(m, m, -cone, rcopy, 1, temp, 1, pinv, m);
}
/** [UpdateRow] */
/**@}*/

// FIXME do we want to keep this in the miniapp?
template<typename MT1, typename MT2>
void checkIdentity(const MT1& a, const MT2& b, const std::string& tag)
{
  constexpr double czero(0.0);
  constexpr double cone(1.0);
  const int nrows = a.rows();
  const int ncols = a.cols();
  double error    = czero;
  for (int i = 0; i < nrows; ++i)
  {
    for (int j = 0; j < nrows; ++j)
    {
      double e = inner_product_n(a[i], b[j], ncols, czero);
      error += (i == j) ? std::abs(e - cone) : std::abs(e);
    }
  }
  #pragma omp master
  std::cout << tag << " difference from identity (average per element) = " << error / nrows / nrows
            << std::endl;
}

// FIXME do we want to keep this in the miniapp?
template<typename MT1, typename MT2>
void checkDiffCPU(const MT1& a, const MT2& b, const std::string& tag)
{
  const int nrows = a.rows();
  const int ncols = a.cols();
  constexpr double czero(0.0);
  double error = czero;
  for (int i = 0; i < nrows; ++i)
    for (int j = 0; j < ncols; ++j)
      error += std::abs(static_cast<double>(a(i, j) - b(i, j)));

  #pragma omp master
  std::cout << tag << " difference between matrices (average per element) = " << error / nrows / nrows
            << std::endl;
}

template<>
class DeterminantDeviceImp<DDT::CPU>
  : public DeterminantDevice<DeterminantDeviceImp<DDT::CPU>>
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
}
} // namespace qmcplusplus

#endif
