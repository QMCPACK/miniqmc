//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2019 QMCPACK developers.
//
// File developed by:
//
// File created by: Jeongnim Kim, jeongnim.kim@intel.com, Intel Corp.
//////////////////////////////////////////////////////////////////////////////////////
// -*- C++ -*-
/** @file algorithm.hpp
 *
 * SIMD version of functions in algorithm
 */
#ifndef QMCPLUSPLUS_SIMD_ALGORITHM_HPP
#define QMCPLUSPLUS_SIMD_ALGORITHM_HPP

namespace qmcplusplus
{
namespace simd
{
template<typename T1, typename T2>
inline T2 accumulate_n(const T1* restrict in, size_t n, T2 res)
{
#pragma omp simd reduction(+ : res)
  for (int i = 0; i < n; ++i)
    res += in[i];
  return res;
}

/** transpose of A(m,n) to B(n,m)
     * @param A starting address, A(m,lda)
     * @param m number of A rows
     * @param lda stride of A's row
     * @param B starting address B(n,ldb)
     * @param n number of B rows
     * @param ldb stride of B's row
     *
     * Blas-like interface
     */
template<typename T, typename TO>
inline void transpose(const T* restrict A, size_t m, size_t lda, TO* restrict B, size_t n, size_t ldb)
{
  for (size_t i = 0; i < n; ++i)
    for (size_t j = 0; j < m; ++j)
      B[i * ldb + j] = A[j * lda + i];
}

/** dot product
     * @param a starting address of an array of type T
     * @param b starting address of an array of type T
     * @param n size
     * @param res initial value, default=0.0
     * @return  \f$ res = \sum_i a[i] b[i]\f$
     *
     * same as inner_product(a,a+n,b,0.0)
     * The arguments of this inline function are different from BLAS::dot
     * This is more efficient than BLAS::dot due to the function overhead,
     * if a compiler knows how to inline.
     */
template<typename T>
inline T dot(const T* restrict a, const T* restrict b, int n, T res = T())
{
  for (int i = 0; i < n; i++)
    res += a[i] * b[i];
  return res;
}

template<class T, unsigned D>
inline TinyVector<T, D> dot(const T* a, const TinyVector<T, D>* b, int n)
{
  TinyVector<T, D> res;
  for (int i = 0; i < n; i++)
    res += a[i] * b[i];
  return res;
}

/** copy function using memcpy
     * @param target starting address of the target
     * @param source starting address of the source
     * @param n size of the data to copy
     */
template<typename T>
inline void copy(T* restrict target, const T* restrict source, size_t n)
{
  memcpy(target, source, sizeof(T) * n);
}

} // namespace simd
} // namespace qmcplusplus
#endif
