////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by:
//
// File created by:
// Jeongnim Kim, jeongnim.kim@intel.com,
//    Intel Corp.
////////////////////////////////////////////////////////////////////////////////
// -*- C++ -*-
/** @file PosTransformer.h
 * @brief Support funtions to handle position type data manged by soa
 */
#ifndef QMCPLUSPLUS_SOA_FAST_PARTICLE_OPERATORS_H
#define QMCPLUSPLUS_SOA_FAST_PARTICLE_OPERATORS_H

namespace qmcplusplus
{
/** General conversion function from AoS[nrows][ncols] to SoA[ncols][ldb]
 * @param nrows the first dimension
 * @param ncols the second dimension
 * @param iptr input pointer
 * @param lda stride of iptr
 * @param out output pointer
 * @param lda strided of out
 *
 * Modeled after blas/lapack for lda/ldb
 */
template<typename T1, typename T2>
void PosAoS2SoA(int nrows, int ncols, const T1* restrict iptr, int lda, T2* restrict out, int ldb)
{
  T2* restrict x = out;
  T2* restrict y = out + ldb;
  T2* restrict z = out + 2 * ldb;
#pragma omp simd aligned(x, y, z)
  for (int i = 0; i < nrows; ++i)
  {
    x[i] = iptr[i * ncols];     // x[i]=in[i][0];
    y[i] = iptr[i * ncols + 1]; // y[i]=in[i][1];
    z[i] = iptr[i * ncols + 2]; // z[i]=in[i][2];
  }
}

/** General conversion function from SoA[ncols][ldb] to AoS[nrows][ncols]
 * @param nrows the first dimension
 * @param ncols the second dimension
 * @param iptr input pointer
 * @param lda stride of iptr
 * @param out output pointer
 * @param lda strided of out
 *
 * Modeled after blas/lapack for lda/ldb
 */
template<typename T1, typename T2>
void PosSoA2AoS(int nrows, int ncols, const T1* restrict iptr, int lda, T2* restrict out, int ldb)
{
  const T1* restrict x = iptr;
  const T1* restrict y = iptr + lda;
  const T1* restrict z = iptr + 2 * lda;
#pragma omp simd aligned(x, y, z)
  for (int i = 0; i < nrows; ++i)
  {
    out[i * ldb]     = x[i]; // out[i][0]=x[i];
    out[i * ldb + 1] = y[i]; // out[i][1]=y[i];
    out[i * ldb + 2] = z[i]; // out[i][2]=z[i];
  }
}

} // namespace qmcplusplus
#endif
