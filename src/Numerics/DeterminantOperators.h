//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License. See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by:
// Ken Esler, kpesler@gmail.com,
//    University of Illinois at Urbana-Champaign
// Miguel Morales, moralessilva2@llnl.gov,
//    Lawrence Livermore National Laboratory
// Jeongnim Kim, jeongnim.kim@gmail.com,
//    University of Illinois at Urbana-Champaign
// Jeremy McMinnis, jmcminis@gmail.com,
//    University of Illinois at Urbana-Champaign
// Mark A. Berrill, berrillma@ornl.gov,
//    Oak Ridge National Laboratory
//
// File created by:
// Jeongnim Kim, jeongnim.kim@gmail.com,
//    University of Illinois at Urbana-Champaign
//////////////////////////////////////////////////////////////////////////////////////

/** @file DeterminantOperators.h
 * @brief Define determinant operators
 */
#ifndef OHMMS_NUMERIC_DETERMINANT_H
#define OHMMS_NUMERIC_DETERMINANT_H

#include <algorithm>
#include <Numerics/OhmmsPETE/TinyVector.h>
#include <Numerics/OhmmsPETE/OhmmsVector.h>
#include <Numerics/OhmmsPETE/OhmmsMatrix.h>
#include <Numerics/OhmmsBlas.h>

namespace qmcplusplus
{

/** LU factorization of double */
inline void LUFactorization(int n, int m, double *QMC_RESTRICT a, int n0,
                            int *QMC_RESTRICT piv)
{
  int status;
  dgetrf(n, m, a, n0, piv, status);
}

/** LU factorization of float */
inline void LUFactorization(int n, int m, float *QMC_RESTRICT a, const int &n0,
                            int *QMC_RESTRICT piv)
{
  int status;
  sgetrf(n, m, a, n0, piv, status);
}

/** LU factorization of std::complex<double> */
inline void LUFactorization(int n, int m, std::complex<double> *QMC_RESTRICT a,
                            int n0, int *QMC_RESTRICT piv)
{
  int status;
  zgetrf(n, m, a, n0, piv, status);
}

/** LU factorization of complex<float> */
inline void LUFactorization(int n, int m, std::complex<float> *QMC_RESTRICT a,
                            int n0, int *QMC_RESTRICT piv)
{
  int status;
  cgetrf(n, m, a, n0, piv, status);
}

/** Inversion of a double matrix after LU factorization*/
inline void InvertLU(int n, double *QMC_RESTRICT a, int n0, int *QMC_RESTRICT piv,
                     double *QMC_RESTRICT work, int n1)
{
  int status;
  dgetri(n, a, n0, piv, work, n1, status);
}

/** Inversion of a float matrix after LU factorization*/
inline void InvertLU(const int &n, float *QMC_RESTRICT a, const int &n0,
                     int *QMC_RESTRICT piv, float *QMC_RESTRICT work, const int &n1)
{
  int status;
  sgetri(n, a, n0, piv, work, n1, status);
}

/** Inversion of a std::complex<double> matrix after LU factorization*/
inline void InvertLU(int n, std::complex<double> *QMC_RESTRICT a, int n0,
                     int *QMC_RESTRICT piv, std::complex<double> *QMC_RESTRICT work,
                     int n1)
{
  int status;
  zgetri(n, a, n0, piv, work, n1, status);
}

/** Inversion of a complex<float> matrix after LU factorization*/
inline void InvertLU(int n, std::complex<float> *QMC_RESTRICT a, int n0,
                     int *QMC_RESTRICT piv, std::complex<float> *QMC_RESTRICT work,
                     int n1)
{
  int status;
  cgetri(n, a, n0, piv, work, n1, status);
}

template <class T>
inline T InvertWithLog(T *QMC_RESTRICT x, int n, int m, T *QMC_RESTRICT work,
                       int *QMC_RESTRICT pivot, T &phase)
{
  T logdet(0.0);
  LUFactorization(n, m, x, n, pivot);
  int sign_det = 1;
  for (int i = 0; i < n; i++)
  {
    sign_det *= (pivot[i] == i + 1) ? 1 : -1;
    sign_det *= (x[i * m + i] > 0) ? 1 : -1;
    logdet += std::log(std::abs(x[i * m + i]));
  }
  InvertLU(n, x, n, pivot, work, n);
  phase = (sign_det > 0) ? 0.0 : M_PI;
  return logdet;
}

template <class T>
inline T InvertWithLog(std::complex<T> *QMC_RESTRICT x, int n, int m,
                       std::complex<T> *QMC_RESTRICT work, int *QMC_RESTRICT pivot,
                       T &phase)
{
  T logdet(0.0);
  LUFactorization(n, m, x, n, pivot);
  phase = 0.0;
  for (int i = 0; i < n; i++)
  {
    int ii = i * m + i;
    phase += std::arg(x[ii]);
    if (pivot[i] != i + 1) phase += M_PI;
    logdet +=
        std::log(x[ii].real() * x[ii].real() + x[ii].imag() * x[ii].imag());
    // slightly smaller error with the following
    //        logdet+=2.0*std::log(std::abs(x[ii]);
  }
  InvertLU(n, x, n, pivot, work, n);
  const T one_over_2pi = 1.0 / TWOPI;
  phase -= std::floor(phase * one_over_2pi) * TWOPI;
  return 0.5 * logdet;
}

/** invert a matrix
 * \param M a matrix to be inverted
 * \param getdet bool, if true, calculate the determinant
 * \return the determinant
 */
template <class MatrixA>
inline typename MatrixA::value_type invert_matrix(MatrixA &M,
                                                  bool getdet = true)
{
  typedef typename MatrixA::value_type value_type;
  const int n = M.rows();
  int pivot[n];
  value_type work[n];
  LUFactorization(n, n, M.data(), n, pivot);
  value_type det0 = 1.0;
  if (getdet)
  // calculate determinant
  {
    int sign = 1;
    for (int i = 0; i < n; ++i)
    {
      if (pivot[i] != i + 1) sign *= -1;
      det0 *= M(i, i);
    }
    det0 *= static_cast<value_type>(sign);
  }
  InvertLU(n, M.data(), n, pivot, work, n);
  return det0;
}
}
#endif
