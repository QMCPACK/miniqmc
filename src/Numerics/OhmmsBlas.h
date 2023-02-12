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
////////////////////////////////////////////////////////////////////////////////

#ifndef QMCPLUSPLUS_NUMERIC_BLAS_H
#define QMCPLUSPLUS_NUMERIC_BLAS_H

// generic header for blas routines
#include "Numerics/Blasf.h"

/** Interfaces to blas library
 *
 *   static data members to facilitate /Fortran blas interface
 *   static member functions to use blas functions
 *   - inline static void axpy
 *   - inline static double norm2
 *   - inline static float norm2
 *   - inline static void symv
 *   - inline static double dot
 *   - inline static float dot
 *
 *  Arguments (float/double/complex\<float\>/complex\<double\>) determine
 *  which BLAS routines are actually used.
 *  Note that symv can be call in many ways.
 */

// clang-format off
struct BLAS
{

  static const int INCX     = 1;
  static const int INCY     = 1;
  static const char UPLO    = 'L';
  static const char TRANS   = 'T';
  static const char NOTRANS = 'N';
  static const float sone;
  static const float szero;
  static const double done;
  static const double dzero;
  static const std::complex<float> cone;
  static const std::complex<float> czero;
  static const std::complex<double> zone;
  static const std::complex<double> zzero;

  inline static void axpy(int n, double x, const double *a, double *b)
  {
    daxpy(n, x, a, INCX, b, INCY);
  }

  inline static void axpy(int n, double x, const double *a, int incx, double *b,
                          int incy)
  {
    daxpy(n, x, a, incx, b, incy);
  }

  inline static void axpy(int n, const double *a, double *b)
  {
    daxpy(n, done, a, INCX, b, INCY);
  }

  inline static void axpy(int n, float x, const float *a, int incx, float *b,
                          int incy)
  {
    saxpy(n, x, a, incx, b, incy);
  }

  inline static void axpy(int n, const std::complex<float> x,
                          const std::complex<float> *a, int incx,
                          std::complex<float> *b, int incy)
  {
    caxpy(n, x, a, incx, b, incy);
  }

  inline static void axpy(int n, const std::complex<double> x,
                          const std::complex<double> *a, int incx,
                          std::complex<double> *b, int incy)
  {
    zaxpy(n, x, a, incx, b, incy);
  }

  inline static double norm2(int n, const double *a, int incx = 1)
  {
    return dnrm2(n, a, incx);
  }

  inline static double norm2(int n, const std::complex<double> *a, int incx = 1)
  {
    return dznrm2(n, a, incx);
  }

  inline static float norm2(int n, const float *a, int incx = 1)
  {
    return snrm2(n, a, incx);
  }

  inline static void scal(int n, double alpha, double *x)
  {
    dscal(n, alpha, x, INCX);
  }

  // inline static
  // void gemv(char trans, int n, int m, const double* amat, const double* x,
  // double* y) {
  //  dgemv(trans, n, m, done, amat, n, x, INCX, dzero, y, INCY);
  //}
  inline static void gemv(int n, int m, const double *restrict amat,
                          const double *restrict x, double *restrict y)
  {
    dgemv(NOTRANS, m, n, done, amat, m, x, INCX, dzero, y, INCY);
  }

  inline static void gemv(int n, int m, const float *restrict amat,
                          const float *restrict x, float *restrict y)
  {
    sgemv(NOTRANS, m, n, done, amat, m, x, INCX, dzero, y, INCY);
  }

  inline static void gemv_trans(int n, int m, const double *restrict amat,
                                const double *restrict x, double *restrict y)
  {
    dgemv(TRANS, m, n, done, amat, m, x, INCX, dzero, y, INCY);
  }

  inline static void gemv_trans(int n, int m, const float *restrict amat,
                                const float *restrict x, float *restrict y)
  {
    sgemv(TRANS, m, n, done, amat, m, x, INCX, dzero, y, INCY);
  }

  inline static void gemv_trans(int n, int m,
                                const std::complex<double> *restrict amat,
                                const std::complex<double> *restrict x,
                                std::complex<double> *restrict y)
  {
    zgemv(TRANS, m, n, done, amat, m, x, INCX, dzero, y, INCY);
  }

  inline static void gemv_trans(int n, int m,
                                const std::complex<float> *restrict amat,
                                const std::complex<float> *restrict x,
                                std::complex<float> *restrict y)
  {
    cgemv(TRANS, m, n, done, amat, m, x, INCX, dzero, y, INCY);
  }

  inline static void gemv(int n, int m,
                          const std::complex<double> *restrict amat,
                          const std::complex<double> *restrict x,
                          std::complex<double> *restrict y)
  {
    zgemv(NOTRANS, m, n, zone, amat, m, x, INCX, zzero, y, INCY);
  }

  inline static void gemv(char trans_in, int n, int m, double alpha,
                          const double *restrict amat, int lda, const double *x,
                          int incx, double beta, double *y, int incy)
  {
    dgemv(trans_in, n, m, alpha, amat, lda, x, incx, beta, y, incy);
  }

  inline static void gemv(char trans_in, int n, int m, float alpha,
                          const float *restrict amat, int lda, const float *x,
                          int incx, float beta, float *y, int incy)
  {
    sgemv(trans_in, n, m, alpha, amat, lda, x, incx, beta, y, incy);
  }

  inline static void gemv(char trans_in, int n, int m,
                          const std::complex<double> &alpha,
                          const std::complex<double> *restrict amat, int lda,
                          const std::complex<double> *restrict x, int incx,
                          const std::complex<double> &beta,
                          std::complex<double> *y, int incy)
  {
    zgemv(trans_in, n, m, alpha, amat, lda, x, incx, beta, y, incy);
  }

  inline static void gemv(char trans_in, int n, int m,
                          const std::complex<float> &alpha,
                          const std::complex<float> *restrict amat, int lda,
                          const std::complex<float> *restrict x, int incx,
                          const std::complex<float> &beta,
                          std::complex<float> *y, int incy)
  {
    cgemv(trans_in, n, m, alpha, amat, lda, x, incx, beta, y, incy);
  }

#if defined(HAVE_MKL)
  inline static void gemv(char trans_in, int n, int m,
                          const std::complex<double> &alpha,
                          const double *restrict amat, int lda,
                          const std::complex<double> *restrict x, int incx,
                          const std::complex<double> &beta,
                          std::complex<double> *y, int incy)
  {
    dzgemv(trans_in, n, m, alpha, amat, lda, x, incx, beta, y, incy);
  }

  inline static void gemv(char trans_in, int n, int m,
                          const std::complex<float> &alpha,
                          const float *restrict amat, int lda,
                          const std::complex<float> *restrict x, int incx,
                          const std::complex<float> &beta,
                          std::complex<float> *y, int incy)
  {
    scgemv(trans_in, n, m, alpha, amat, lda, x, incx, beta, y, incy);
  }
#endif

  inline static void gemm(char Atrans, char Btrans, int M, int N, int K,
                          double alpha, const double *A, int lda,
                          const double *restrict B, int ldb, double beta,
                          double *restrict C, int ldc)
  {
    dgemm(Atrans, Btrans, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
  }

  inline static void gemm(char Atrans, char Btrans, int M, int N, int K,
                          float alpha, const float *A, int lda,
                          const float *restrict B, int ldb, float beta,
                          float *restrict C, int ldc)
  {
    sgemm(Atrans, Btrans, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
  }

  inline static void gemm(char Atrans, char Btrans, int M, int N, int K,
                          std::complex<double> alpha,
                          const std::complex<double> *A, int lda,
                          const std::complex<double> *restrict B, int ldb,
                          std::complex<double> beta,
                          std::complex<double> *restrict C, int ldc)
  {
    zgemm(Atrans, Btrans, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
  }

  inline static void gemm(char Atrans, char Btrans, int M, int N, int K,
                          std::complex<float> alpha,
                          const std::complex<float> *A, int lda,
                          const std::complex<float> *restrict B, int ldb,
                          std::complex<float> beta,
                          std::complex<float> *restrict C, int ldc)
  {
    cgemm(Atrans, Btrans, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
  }

  template <typename T>
  inline static T dot(int n, const T *restrict a, const T *restrict b)
  {
    T res = T(0);
    for (int i = 0; i < n; ++i)
      res += a[i] * b[i];
    return res;
  }

  template <typename T>
  inline static std::complex<T> dot(int n, const std::complex<T> *restrict a,
                                    const T *restrict b)
  {
    std::complex<T> res = T(0);
    for (int i = 0; i < n; ++i)
      res += a[i] * b[i];
    return res;
  }

  template <typename T>
  inline static std::complex<T> dot(int n, const std::complex<T> *restrict a,
                                    const std::complex<T> *restrict b)
  {
    std::complex<T> res = 0.0;
    for (int i = 0; i < n; ++i)
      res += a[i] * b[i];
    return res;
  }

  template <typename T>
  inline static std::complex<T> dot(int n, const T *restrict a,
                                    const std::complex<T> *restrict b)
  {
    std::complex<T> res = 0.0;
    for (int i = 0; i < n; ++i)
      res += a[i] * b[i];
    return res;
  }

  template <typename T>
  inline static void copy(int n, const T *restrict a, T *restrict b)
  {
    memcpy(b, a, sizeof(T) * n);
  }

  /** copy using memcpy(target,source,size)
   * @param target starting address of the targe
   * @param source starting address of the source
   * @param number of elements to copy
   */
  template <typename T>
  inline static void copy(T *restrict target, const T *restrict source, int n)
  {
    memcpy(target, source, sizeof(T) * n);
  }

  template <typename T>
  inline static void copy(int n, const std::complex<T> *restrict a,
                          T *restrict b)
  {
    for (int i = 0; i < n; ++i)
      b[i]     = a[i].real();
  }

  template <typename T>
  inline static void copy(int n, const T *restrict a,
                          std::complex<T> *restrict b)
  {
    for (int i = 0; i < n; ++i)
      b[i]     = a[i];
  }

  template <typename T>
  inline static void copy(int n, const T *restrict x, int incx, T *restrict y,
                          int incy)
  {
    const int xmax = incx * n;
    for (int ic = 0, jc = 0; ic < xmax; ic += incx, jc += incy)
      y[jc] = x[ic];
  }

  inline static void ger(int m, int n, double alpha, const double *x, int incx,
                         const double *y, int incy, double *a, int lda)
  {
    dger(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
  }

  inline static void ger(int m, int n, float alpha, const float *x, int incx,
                         const float *y, int incy, float *a, int lda)
  {
    sger(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
  }

  inline static void ger(int m, int n, const std::complex<double> &alpha,
                         const std::complex<double> *x, int incx,
                         const std::complex<double> *y, int incy,
                         std::complex<double> *a, int lda)
  {
    zgeru(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
  }

  inline static void ger(int m, int n, const std::complex<float> &alpha,
                         const std::complex<float> *x, int incx,
                         const std::complex<float> *y, int incy,
                         std::complex<float> *a, int lda)
  {
    cgeru(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
  }
};

namespace LAPACK
{
  inline void getrf(int n, int m, float* a, int lda, int* piv, int& status)
  {
    sgetrf(n, m, a, lda, piv, status);
  }

  inline void getrf(int n, int m, std::complex<float>* a, int lda, int* piv, int& status)
  {
    cgetrf(n, m, a, lda, piv, status);
  }

  inline void getrf(int n, int m, double* a, int lda, int* piv, int& status)
  {
    dgetrf(n, m, a, lda, piv, status);
  }

  inline void getrf(int n, int m, std::complex<double>* a, int lda, int* piv, int& status)
  {
    zgetrf(n, m, a, lda, piv, status);
  }

  inline void getri(int n, float* a, int lda, int* piv, float* work, int& lwork, int& status)
  {
    sgetri(n, a, lda, piv, work, lwork, status);
  }

  inline void getri(int n, std::complex<float>* a, int lda, int* piv, std::complex<float>* work, int& lwork, int& status)
  {
    cgetri(n, a, lda, piv, work, lwork, status);
  }

  inline void getri(int n, double* a, int lda, int* piv, double* work, int& lwork, int& status)
  {
    dgetri(n, a, lda, piv, work, lwork, status);
  }

  inline void getri(int n, std::complex<double>* a, int lda, int* piv, std::complex<double>* work, int& lwork, int& status)
  {
    zgetri(n, a, lda, piv, work, lwork, status);
  }

};
// clang-format on
#endif // OHMMS_BLAS_H
