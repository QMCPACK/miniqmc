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

#ifndef QMCPLUSPLUS_DETERMINANT_H
#define QMCPLUSPLUS_DETERMINANT_H

#include <Kokkos_Core.hpp>
#include <impl/Kokkos_Timer.hpp>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#ifdef KOKKOS_ENABLE_CUDA
#include "cublas_v2.h"
#endif

#include "QMCWaveFunctions/WaveFunctionComponent.h"
//#include "Utilities/RandomGenerator.h"

#define dgetrf dgetrf_
#define sgetrf sgetrf_
#define zgetrf zgetrf_
#define cgetrf cgetrf_
#define dgetri dgetri_
#define sgetri sgetri_
#define zgetri zgetri_
#define cgetri cgetri_
#define dgemv dgemv_
#define sgemv sgemv_
#define zgemv zgemv_
#define cgemv cgemv_
#define dger dger_
#define sger sger_
#define zgeru zgeru_
#define cgeru cgeru_
#define dgemm dgemm_
#define sgemm sgemm_
#define zgemm zgemm_
#define cgemm cgemm_


extern "C" {

void dgetrf(const int &n, const int &m, double *a, const int &n0, int *piv,
            int &st);
void sgetrf(const int &n, const int &m, float *a, const int &n0, int *piv,
            int &st); 
void zgetrf(const int &n, const int &m, std::complex<double> *a, const int &n0,
            int *piv, int &st);
void cgetrf(const int &n, const int &m, std::complex<float> *a, const int &n0,
            int *piv, int &st);

void dgetri(const int &n, double *a, const int &n0, int *piv, double *work,
            const int &lwork, int &st);
void sgetri(const int &n, float *a, const int &n0, int *piv, float *work,
            const int &lwork, int &st);
void zgetri(const int &n, std::complex<double> *a, const int &n0, int *piv,
            std::complex<double> *work, const int &lwork, int &st);
void cgetri(const int &n, std::complex<float> *a, const int &n0, int *piv,
            std::complex<float> *work, const int &lwork, int &st);

void dgemv(const char &trans, const int &nr, const int &nc, const double &alpha,
           const double *amat, const int &lda, const double *bv,
           const int &incx, const double &beta, double *cv, const int &incy);
void sgemv(const char &trans, const int &nr, const int &nc, const float &alpha,
           const float *amat, const int &lda, const float *bv, const int &incx,
           const float &beta, float *cv, const int &incy);
void zgemv(const char &trans, const int &nr, const int &nc,
           const std::complex<double> &alpha, const std::complex<double> *amat,
           const int &lda, const std::complex<double> *bv, const int &incx,
           const std::complex<double> &beta, std::complex<double> *cv,
           const int &incy);
void cgemv(const char &trans, const int &nr, const int &nc,
           const std::complex<float> &alpha, const std::complex<float> *amat,
           const int &lda, const std::complex<float> *bv, const int &incx,
           const std::complex<float> &beta, std::complex<float> *cv,
           const int &incy);

void dger(const int *m, const int *n, const double *alpha, const double *x,
          const int *incx, const double *y, const int *incy, double *a,
          const int *lda);
void sger(const int *m, const int *n, const float *alpha, const float *x,
          const int *incx, const float *y, const int *incy, float *a,
          const int *lda);
void zgeru(const int *m, const int *n, const std::complex<double>* alpha,
           const std::complex<double> *x, const int *incx,
           const std::complex<double> *y, const int *incy,
           std::complex<double> *a, const int *lda);
void cgeru(const int *m, const int *n, const std::complex<float>* alpha,
           const std::complex<float> *x, const int *incx,
           const std::complex<float> *y, const int *incy,
           std::complex<float> *a, const int *lda);
void dgemm(const char &, const char &, const int &, const int &, const int &,
           const double &, const double *, const int &, const double *,
           const int &, const double &, double *, const int &);

void sgemm(const char &, const char &, const int &, const int &, const int &,
           const float &, const float *, const int &, const float *,
           const int &, const float &, float *, const int &);

void zgemm(const char &, const char &, const int &, const int &, const int &,
           const std::complex<double> &, const std::complex<double> *,
           const int &, const std::complex<double> *, const int &,
           const std::complex<double> &, std::complex<double> *, const int &);

void cgemm(const char &, const char &, const int &, const int &, const int &,
           const std::complex<float> &, const std::complex<float> *,
           const int &, const std::complex<float> *, const int &,
           const std::complex<float> &, std::complex<float> *, const int &);
}


// note that LeftLayout is column-major, and is the format that
// cusolver and likely cuBlas will require

namespace qmcplusplus
{
// assuming both views have the same dimensionality and 
// that the types they hold can assigned between
// CURRENTLY ONLY IMPLEMENTED UP TO RANK 5 tensors
template<class ViewType1, class ViewType2>
void elementWiseCopy(ViewType1 destination, ViewType2 source, 
		     typename std::enable_if<ViewType1::rank==1>::type* = 0,
		     typename std::enable_if<ViewType2::rank==1>::type* = 0) {
  for (int i = 0; i < ViewType1::rank; i++) {
    assert(destination.extent(i) == source.extent(i));
  }
  Kokkos::parallel_for("elementWiseCopy::copy_elements_rk1", 
		       destination.extent(0),
		       KOKKOS_LAMBDA(const int& i0) {
			 destination(i0) = source(i0);
		       });
  Kokkos::fence();
}
template<class ViewType1, class ViewType2>
void elementWiseCopy(ViewType1 destination, ViewType2 source, 
		     typename std::enable_if<ViewType1::rank==2>::type* = 0,
		     typename std::enable_if<ViewType2::rank==2>::type* = 0) {
  for (int i = 0; i < ViewType1::rank; i++) {
    assert(destination.extent(i) == source.extent(i));
  }
  Kokkos::parallel_for("elementWiseCopy::copy_elements_rk2", 
		       Kokkos::MDRangePolicy<Kokkos::Rank<2,Kokkos::Iterate::Left> >({0,0}, {destination.extent(0),destination.extent(1)}), 
		       KOKKOS_LAMBDA(const int& i0, const int& i1) {
			 destination(i0, i1) = source(i0, i1);
		       });
  Kokkos::fence();
}
 template<class ViewType1, class ViewType2>
void elementWiseCopyTrans(ViewType1 destination, ViewType2 source, 
			  typename std::enable_if<ViewType1::rank==2>::type* = 0,
			  typename std::enable_if<ViewType2::rank==2>::type* = 0) {
   assert(destination.extent(0) == source.extent(1));
   assert(destination.extent(1) == source.extent(0));

   Kokkos::parallel_for("elementWiseCopy::copy_elements_rk2", 
		       Kokkos::MDRangePolicy<Kokkos::Rank<2,Kokkos::Iterate::Left> >({0,0}, {destination.extent(0),destination.extent(1)}), 
		       KOKKOS_LAMBDA(const int& i0, const int& i1) {
			 destination(i0, i1) = source(i1, i0);
		       });
   Kokkos::fence();
}
template<class ViewType1, class ViewType2>
void elementWiseCopy(ViewType1 destination, ViewType2 source, 
		     typename std::enable_if<ViewType1::rank==3>::type* = 0,
		     typename std::enable_if<ViewType2::rank==3>::type* = 0) {
  for (int i = 0; i < ViewType1::rank; i++) {
    assert(destination.extent(i) == source.extent(i));
  }
  Kokkos::parallel_for("elementWiseCopy::copy_elements_rk3", 
			 Kokkos::MDRangePolicy<Kokkos::Rank<3,Kokkos::Iterate::Left> >({0,0,0}, {destination.extent(0),destination.extent(1),destination.extent(2)}), 
			 KOKKOS_LAMBDA(const int& i0, const int& i1, const int& i2) {
			   destination(i0, i1, i2) = source(i0, i1, i2);
			 });
  Kokkos::fence();
}
template<class ViewType1, class ViewType2>
void elementWiseCopy(ViewType1 destination, ViewType2 source, 
		     typename std::enable_if<ViewType1::rank==4>::type* = 0,
		     typename std::enable_if<ViewType2::rank==4>::type* = 0) {
  for (int i = 0; i < ViewType1::rank; i++) {
    assert(destination.extent(i) == source.extent(i));
  }
  Kokkos::parallel_for("elementWiseCopy::copy_elements_rk4", 
		       Kokkos::MDRangePolicy<Kokkos::Rank<4,Kokkos::Iterate::Left> >({0,0,0,0}, {destination.extent(0),destination.extent(1),destination.extent(2),destination.extent(3)}), 
		       KOKKOS_LAMBDA(const int& i0, const int& i1, const int& i2, const int& i3) {
			 destination(i0, i1, i2, i3) = source(i0, i1, i2, i3);
		       });
  Kokkos::fence();
}
template<class ViewType1, class ViewType2>
void elementWiseCopy(ViewType1 destination, ViewType2 source, 
		     typename std::enable_if<ViewType1::rank==5>::type* = 0,
		     typename std::enable_if<ViewType2::rank==5>::type* = 0) {
  for (int i = 0; i < ViewType1::rank; i++) {
    assert(destination.extent(i) == source.extent(i));
  }
  Kokkos::parallel_for("elementWiseCopy::copy_elements_rk5", 
		       Kokkos::MDRangePolicy<Kokkos::Rank<5,Kokkos::Iterate::Left> >({0,0,0,0,0}, {destination.extent(0),destination.extent(1),destination.extent(2),destination.extent(3),destination.extent(4)}), 
		       KOKKOS_LAMBDA(const int& i0, const int& i1, const int& i2, const int& i3, const int& i4) {
			 destination(i0, i1, i2, i3, i4) = source(i0, i1, i2, i3, i4);
		       });
  Kokkos::fence();
}



template<typename valueType, typename arrayLayout, typename memorySpace>
void checkTemplateParams() {
  static_assert(std::is_same<arrayLayout, Kokkos::LayoutLeft>::value, "Require LayoutLeft Views for the time being to interface with linear algebra libraries");

#ifdef KOKKOS_ENABLE_CUDA
  static_assert(std::is_same<memorySpace, Kokkos::HostSpace>::value || std::is_same<memorySpace, Kokkos::CudaSpace>::value || std::is_same<memorySpace, Kokkos::CudaUVMSpace>::value, "Currently only know about HostSpace, CudaSpace and CudaUVMSpace views");
  static_assert(std::is_same<valueType, float>::value ||
                std::is_same<valueType, double>::value ||
		std::is_same<valueType, Kokkos::complex<float> >::value ||
		std::is_same<valueType, std::complex<float> >::value ||
		std::is_same<valueType, Kokkos::complex<double> >::value ||
		std::is_same<valueType, std::complex<double> >::value ||
		std::is_same<valueType, cuFloatComplex>::value ||
		std::is_same<valueType, cuDoubleComplex>::value, "Currently only support, float, double, and std/Kokkos/ complex<float> or complex<double>, cor cuFloatComplex or cuDoubleComplex");
#else
  static_assert(std::is_same<memorySpace, Kokkos::HostSpace>::value, "For this build of Kokkos, currently understand HostSpace Views, maybe try building with cuda?");
  static_assert(std::is_same<valueType, float>::value ||
                std::is_same<valueType, double>::value ||
		std::is_same<valueType, Kokkos::complex<float> >::value ||
		std::is_same<valueType, std::complex<float> >::value ||
		std::is_same<valueType, Kokkos::complex<double> >::value ||
		std::is_same<valueType, std::complex<double> >::value, "Currently only support, float, double, and std/Kokkos complex<float> or complex<double>");
#endif
}


void getrf_cpu_impl(const int &n, const int& m, float* a, const int &n0, int* piv, int& st) {
  sgetrf(n,m,a,n0,piv,st);
}
void getrf_cpu_impl(const int &n, const int& m, double* a, const int &n0, int* piv, int& st) {
  dgetrf(n,m,a,n0,piv,st);
}
void getrf_cpu_impl(const int &n, const int& m, std::complex<float>* a, const int &n0, int* piv, int& st) {
  cgetrf(n,m,a,n0,piv,st);
}   
void getrf_cpu_impl(const int &n, const int& m, std::complex<double>* a, const int &n0, int* piv, int& st) {
  zgetrf(n,m,a,n0,piv,st);
}   

void getri_cpu_impl(const int &n, float* a, const int &n0, int* piv, float*work, const int& lwork, int &st) {
  sgetri(n,a,n0,piv,work,lwork,st);
}
void getri_cpu_impl(const int &n, double* a, const int &n0, int* piv, double*work, const int& lwork, int &st) {
  dgetri(n,a,n0,piv,work,lwork,st);
}
void getri_cpu_impl(const int &n, std::complex<float>* a, const int &n0, int* piv, std::complex<float>*work, const int& lwork, int &st) {
  cgetri(n,a,n0,piv,work,lwork,st);
}
void getri_cpu_impl(const int &n, std::complex<double>* a, const int &n0, int* piv, std::complex<double>*work, const int& lwork, int &st) {
  zgetri(n,a,n0,piv,work,lwork,st);
}

void gemv_cpu_impl(const char& trans, const int& nr, const int& nc, const float &alpha,
		   const float *amat, const int &lda, const float *bv, const int &incx,
		   const float &beta, float *cv, const int &incy) {
  sgemv(trans, nr, nc, alpha, amat, lda, bv, incx, beta, cv, incy);
}
void gemv_cpu_impl(const char& trans, const int& nr, const int& nc, const double &alpha,
		   const double *amat, const int &lda, const double *bv, const int &incx,
		   const double &beta, double *cv, const int &incy) {
  dgemv(trans, nr, nc, alpha, amat, lda, bv, incx, beta, cv, incy);
}
void gemv_cpu_impl(const char& trans, const int& nr, const int& nc, const std::complex<float> &alpha,
		   const std::complex<float> *amat, const int &lda, const std::complex<float> *bv, const int &incx,
		   const std::complex<float> &beta, std::complex<float> *cv, const int &incy) {
  cgemv(trans, nr, nc, alpha, amat, lda, bv, incx, beta, cv, incy);
}
void gemv_cpu_impl(const char& trans, const int& nr, const int& nc, const std::complex<double> &alpha,
		   const std::complex<double> *amat, const int &lda, const std::complex<double> *bv, const int &incx,
		   const std::complex<double> &beta, std::complex<double> *cv, const int &incy) {
  zgemv(trans, nr, nc, alpha, amat, lda, bv, incx, beta, cv, incy);
}

void ger_cpu_impl(const int& m, const int& n, const float& alpha, const float *x,
		  const int& incx, const float *y, const int& incy, float *a,
		  const int& lda) {
  sger(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
}
void ger_cpu_impl(const int& m, const int& n, const double& alpha, const double *x,
		  const int& incx, const double *y, const int& incy, double *a,
		  const int& lda) {
  dger(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
}
void ger_cpu_impl(const int& m, const int& n, const std::complex<float>& alpha, const std::complex<float> *x,
		  const int& incx, const std::complex<float> *y, const int& incy, std::complex<float> *a,
		  const int& lda) {
  cgeru(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
}
void ger_cpu_impl(const int& m, const int& n, const std::complex<double>& alpha, const std::complex<double> *x,
		  const int& incx, const std::complex<double> *y, const int& incy, std::complex<double> *a,
		  const int& lda) {
  zgeru(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
}
void gemm_cpu_impl(const char& transa, const char& transb, const int& rowsa, const int& columnsb, 
		   const int& columnsa, const float* alpha, const float* a, const int& lda,
		   const float* b, const int& ldb, const float* beta, float* c, const int& ldc) {
  sgemm(transa, transb, rowsa, columnsb, columnsa, *alpha, a, lda, b, ldb, *beta, c, ldc);
}
void gemm_cpu_impl(const char& transa, const char& transb, const int& rowsa, const int& columnsb, 
		   const int& columnsa, const double* alpha, const double* a, const int& lda,
		   const double* b, const int& ldb, const double* beta, double* c, const int& ldc) {
  dgemm(transa, transb, rowsa, columnsb, columnsa, *alpha, a, lda, b, ldb, *beta, c, ldc);
}
void gemm_cpu_impl(const char& transa, const char& transb, const int& rowsa, const int& columnsb, 
		   const int& columnsa, const std::complex<float>* alpha, const std::complex<float>* a, const int& lda,
		   const std::complex<float>* b, const int& ldb, const std::complex<float>* beta, std::complex<float>* c, const int& ldc) {
  cgemm(transa, transb, rowsa, columnsb, columnsa, *alpha, a, lda, b, ldb, *beta, c, ldc);
}
void gemm_cpu_impl(const char& transa, const char& transb, const int& rowsa, const int& columnsb, 
		   const int& columnsa, const std::complex<double> *alpha, const std::complex<double>* a, const int& lda,
		   const std::complex<double>* b, const int& ldb, const std::complex<double>* beta, std::complex<double>* c, const int& ldc) {
  zgemm(transa, transb, rowsa, columnsb, columnsa, *alpha, a, lda, b, ldb, *beta, c, ldc);
}


#ifdef KOKKOS_ENABLE_CUDA
void getrf_gpu_impl(const int &n, float** a, int* piv, int* info, cublasHandle_t& handle) {
  int st = cublasSgetrfBatched(handle, n, a, n, piv, info, 1);
}
void getrf_gpu_impl(const int &n, double** a, int* piv, int* info, cublasHandle_t& handle) {
  int st = cublasDgetrfBatched(handle, n, a, n, piv, info, 1);
}
void getrf_gpu_impl(const int &n, cuFloatComplex** a, int* piv, int* info, cublasHandle_t& handle) {
  int st = cublasCgetrfBatched(handle, n, a, n, piv, info, 1);
}
void getrf_gpu_impl(const int &n, cuDoubleComplex** a, int* piv, int* info, cublasHandle_t& handle) {
  int st = cublasZgetrfBatched(handle, n, a, n, piv, info, 1);
}

void getri_gpu_impl(const int &n, float** a, float**b, int* piv, int* info, cublasHandle_t& handle) {
  int st = cublasSgetriBatched(handle,n,a,n,piv,b,n,info,1);
}
void getri_gpu_impl(const int &n, double** a, double**b, int* piv, int* info, cublasHandle_t& handle) {
  int st = cublasDgetriBatched(handle,n,a,n,piv,b,n,info,1);
}
void getri_gpu_impl(const int &n, cuFloatComplex** a, cuFloatComplex**b, int* piv, int* info, cublasHandle_t& handle) {
  int st = cublasCgetriBatched(handle,n,a,n,piv,b,n,info,1);
}
void getri_gpu_impl(const int &n, cuDoubleComplex** a, cuDoubleComplex**b, int* piv, int* info, cublasHandle_t& handle) {
  int st = cublasZgetriBatched(handle,n,a,n,piv,b,n,info,1);
}

void gemv_gpu_impl(cublasHandle_t& handle, cublasOperation_t trans, const int& nr, const int& nc, const float *alpha,
		   const float *amat, const int &lda, const float *bv, const int &incx,
		   const float *beta, float *cv, const int &incy) {
  cublasSgemv(handle, trans, nr, nc, alpha, amat, lda, bv, incx, beta, cv, incy);
}
void gemv_gpu_impl(cublasHandle_t& handle, cublasOperation_t trans, const int& nr, const int& nc, const double *alpha,
		   const double *amat, const int &lda, const double *bv, const int &incx,
		   const double *beta, double *cv, const int &incy) {
  cublasDgemv(handle, trans, nr, nc, alpha, amat, lda, bv, incx, beta, cv, incy);
}
void gemv_gpu_impl(cublasHandle_t& handle, cublasOperation_t trans, const int& nr, const int& nc, const cuFloatComplex *alpha,
		   const cuFloatComplex *amat, const int &lda, const cuFloatComplex *bv, const int &incx,
		   const cuFloatComplex *beta, cuFloatComplex *cv, const int &incy) {
  cublasCgemv(handle, trans, nr, nc, alpha, amat, lda, bv, incx, beta, cv, incy);
}
void gemv_gpu_impl(cublasHandle_t& handle, cublasOperation_t trans, const int& nr, const int& nc, const cuDoubleComplex *alpha,
		   const cuDoubleComplex *amat, const int &lda, const cuDoubleComplex *bv, const int &incx,
		   const cuDoubleComplex *beta, cuDoubleComplex *cv, const int &incy) {
  cublasZgemv(handle, trans, nr, nc, alpha, amat, lda, bv, incx, beta, cv, incy);
}

void ger_gpu_impl(cublasHandle_t& handle, const int& m, const int& n, const float* alpha, const float *x,
		  const int& incx, const float *y, const int& incy, float *a,
		  const int& lda) {
  cublasSger(handle, m, n, alpha, x, incx, y, incy, a, lda);
}
void ger_gpu_impl(cublasHandle_t& handle, const int& m, const int& n, const double* alpha, const double *x,
		  const int& incx, const double *y, const int& incy, double *a,
		  const int& lda) {
  cublasDger(handle, m, n, alpha, x, incx, y, incy, a, lda);
}
void ger_gpu_impl(cublasHandle_t& handle, const int& m, const int& n, const cuFloatComplex* alpha, const cuFloatComplex *x,
		  const int& incx, const cuFloatComplex *y, const int& incy, cuFloatComplex *a,
		  const int& lda) {
  cublasCgeru(handle, m, n, alpha, x, incx, y, incy, a, lda);
}
void ger_gpu_impl(cublasHandle_t& handle, const int& m, const int& n, const cuDoubleComplex* alpha, const cuDoubleComplex *x,
		  const int& incx, const cuDoubleComplex *y, const int& incy, cuDoubleComplex *a,
		  const int& lda) {
  cublasZgeru(handle, m, n, alpha, x, incx, y, incy, a, lda);
}
void gemm_gpu_impl(cublasHandle_t& handle, cublasOperation_t transa, cublasOperation_t transb,
		   const int& rowsa, const int& columnsb, const int& columnsa, const float *alpha, 
		   const float* a, const int& lda,
		   const float* b, const int& ldb, 
		   const float* beta, float* c, const int& ldc) {
  cublasSgemm(handle, transa, transb, rowsa, columnsb, columnsa, alpha, a, lda, b, ldb, beta, c, ldc);
}
void gemm_gpu_impl(cublasHandle_t& handle, cublasOperation_t transa, cublasOperation_t transb,
		   const int& rowsa, const int& columnsb, const int& columnsa, const double *alpha, 
		   const double* a, const int& lda,
		   const double* b, const int& ldb, 
		   const double* beta, double* c, const int& ldc) {
  cublasDgemm(handle, transa, transb, rowsa, columnsb, columnsa, alpha, a, lda, b, ldb, beta, c, ldc);
}
void gemm_gpu_impl(cublasHandle_t& handle, cublasOperation_t transa, cublasOperation_t transb,
		   const int& rowsa, const int& columnsb, const int& columnsa, const cuFloatComplex *alpha, 
		   const cuFloatComplex* a, const int& lda,
		   const cuFloatComplex* b, const int& ldb, 
		   const cuFloatComplex* beta, cuFloatComplex* c, const int& ldc) {
  cublasCgemm(handle, transa, transb, rowsa, columnsb, columnsa, alpha, a, lda, b, ldb, beta, c, ldc);
}
void gemm_gpu_impl(cublasHandle_t& handle, cublasOperation_t transa, cublasOperation_t transb,
		   const int& rowsa, const int& columnsb, const int& columnsa, const cuDoubleComplex *alpha, 
		   const cuDoubleComplex* a, const int& lda,
		   const cuDoubleComplex* b, const int& ldb, 
		   const cuDoubleComplex* beta, cuDoubleComplex* c, const int& ldc) {
  cublasZgemm(handle, transa, transb, rowsa, columnsb, columnsa, alpha, a, lda, b, ldb, beta, c, ldc);
}
#endif



template<typename valueType, typename arrayLayout, typename memorySpace>
class linalgHelper {
 public:
  using viewType = Kokkos::View<valueType**, arrayLayout, memorySpace>;
  using arrType = Kokkos::View<valueType*, memorySpace>;

private:
  int status;
  Kokkos::View<int*, memorySpace> piv;
  Kokkos::View<valueType*, memorySpace> work;

  double* pointerConverter(double* d) {  return d; }
  float* pointerConverter(float* d) {  return d; }
  std::complex<float>* pointerConverter(std::complex<float>* d) { return d; }
  std::complex<float>* pointerConverter(Kokkos::complex<float>* d) { return (std::complex<float>*)d; }
  std::complex<double>* pointerConverter(std::complex<double>* d) { return d; }
  std::complex<double>* pointerConverter(Kokkos::complex<double>* d) { return (std::complex<double>*)d; }
  
public:
  linalgHelper() {
    //checkTemplateParams<valueType, arrayLayout, memorySpace>();
    Kokkos::resize(piv, 1);
    Kokkos::resize(work, 1);
    status = -1;
  }

  typename Kokkos::View<int*, memorySpace>::HostMirror extractPivot() {
    //typename Kokkos::View<int*, memorySpace>::HostMirror piv_mirror = Kokkos::create_mirror_view(piv);
    auto piv_mirror = Kokkos::create_mirror_view(piv);
    Kokkos::deep_copy(piv_mirror, piv);
    return piv_mirror;
  }

  void getrf(viewType view) {
    int ext = view.extent(0);
    if(piv.extent(0) != ext) {
      Kokkos::resize(piv, ext);
    }
    status = -1;
    getrf_cpu_impl(ext,ext,pointerConverter(view.data()),ext,piv.data(),status);
  }
  // note this assumes that we just used getrf and the pivot is still in piv
  void getri(viewType view) {
    int ext = view.extent(0);
    if(piv.extent(0) != ext) {
      // we're hosed, this will NOT work
      Kokkos::resize(piv,ext);
    }
    /* should work in principle, problem is the code below fails if we are dealing with a complex data type (cast fails)
    // do initial call to find out how big the workspace needs to be
    getri_cpu_impl(ext, pointerConverter(view.data()), ext, piv.data(), pointerConverter(work.data()), -1, status);

    // now check that workspace is sufficient and resize if not
    if(work.extent(0) != static_cast<int>(work(0))) {
      Kokkos::resize(work,static_cast<int>(work(0)));
    }
    */
    if (work.extent(0) < ext*ext) {
      Kokkos::resize(work,ext*ext);
    }

    // now do call to invert matrix
    getri_cpu_impl(ext, pointerConverter(view.data()), ext, piv.data(), pointerConverter(work.data()), work.extent(0), status);
  }
  void invertMatrix(viewType view) {
    // group calls to make sure pivot is not touched
    getrf(view);
    getri(view);
  }
  void gemvTrans(viewType A, arrType x, arrType y, valueType alpha, valueType beta) {
    gemv_cpu_impl('T', A.extent(0), A.extent(1), alpha, pointerConverter(A.data()), A.extent(0), pointerConverter(x.data()),
		  1, beta, pointerConverter(y.data()), 1);
  }
  void gemvConj(viewType A, arrType x, arrType y, valueType alpha, valueType beta) {
    gemv_cpu_impl('C', A.extent(0), A.extent(1), alpha, pointerConverter(A.data()), A.extent(0), pointerConverter(x.data()),
		  1, beta, pointerConverter(y.data()), 1);
  }
  void gemvNorm(viewType A, arrType x, arrType y, valueType alpha, valueType beta) {
    gemv_cpu_impl('N', A.extent(0), A.extent(1), alpha, pointerConverter(A.data()), A.extent(0), pointerConverter(x.data()),
		  1, beta, pointerConverter(y.data()), 1);
  }
  void ger(viewType A, arrType x, arrType y, valueType alpha) {
    ger_cpu_impl(A.extent(0), A.extent(1), alpha, pointerConverter(x.data()), 1, pointerConverter(y.data()), 1,
		 pointerConverter(A.data()), A.extent(0));
  }
  void gemmNN(viewType A, viewType B, viewType C, valueType alpha, valueType beta) {
    gemm_cpu_impl('N', 'N', A.extent(0), B.extent(1), A.extent(1), pointerConverter(&alpha),
		  pointerConverter(A.data()), A.extent(0), pointerConverter(B.data()), B.extent(0),
		  pointerConverter(&beta), pointerConverter(C.data()), C.extent(0));
  }
  void gemmNT(viewType A, viewType B, viewType C, valueType alpha, valueType beta) {
    gemm_cpu_impl('N', 'T', A.extent(0), B.extent(1), A.extent(1), pointerConverter(&alpha),
		  pointerConverter(A.data()), A.extent(0), pointerConverter(B.data()), B.extent(0),
		  pointerConverter(&beta), pointerConverter(C.data()), C.extent(0));
  }
  void gemmNC(viewType A, viewType B, viewType C, valueType alpha, valueType beta) {
    gemm_cpu_impl('N', 'C', A.extent(0), B.extent(1), A.extent(1), pointerConverter(&alpha),
		  pointerConverter(A.data()), A.extent(0), pointerConverter(B.data()), B.extent(0),
		  pointerConverter(&beta), pointerConverter(C.data()), C.extent(0));
  }
  void gemmCN(viewType A, viewType B, viewType C, valueType alpha, valueType beta) {
    gemm_cpu_impl('C', 'N', A.extent(0), B.extent(1), A.extent(1), pointerConverter(&alpha),
		  pointerConverter(A.data()), A.extent(0), pointerConverter(B.data()), B.extent(0),
		  pointerConverter(&beta), pointerConverter(C.data()), C.extent(0));
  }
  void gemmCT(viewType A, viewType B, viewType C, valueType alpha, valueType beta) {
    gemm_cpu_impl('C', 'T', A.extent(0), B.extent(1), A.extent(1), pointerConverter(&alpha),
		  pointerConverter(A.data()), A.extent(0), pointerConverter(B.data()), B.extent(0),
		  pointerConverter(&beta), pointerConverter(C.data()), C.extent(0));
  }
  void gemmCC(viewType A, viewType B, viewType C, valueType alpha, valueType beta) {
    gemm_cpu_impl('C', 'C', A.extent(0), B.extent(1), A.extent(1), pointerConverter(&alpha),
		  pointerConverter(A.data()), A.extent(0), pointerConverter(B.data()), B.extent(0),
		  pointerConverter(&beta), pointerConverter(C.data()), C.extent(0));
  }
  void gemmTN(viewType A, viewType B, viewType C, valueType alpha, valueType beta) {
    gemm_cpu_impl('T', 'N', A.extent(0), B.extent(1), A.extent(1), pointerConverter(&alpha),
		  pointerConverter(A.data()), A.extent(0), pointerConverter(B.data()), B.extent(0),
		  pointerConverter(&beta), pointerConverter(C.data()), C.extent(0));
  }
  void gemmTT(viewType A, viewType B, viewType C, valueType alpha, valueType beta) {
    gemm_cpu_impl('T', 'T', A.extent(0), B.extent(1), A.extent(1), pointerConverter(&alpha),
		  pointerConverter(A.data()), A.extent(0), pointerConverter(B.data()), B.extent(0),
		  pointerConverter(&beta), pointerConverter(C.data()), C.extent(0));
  }
  void gemmTC(viewType A, viewType B, viewType C, valueType alpha, valueType beta) {
    gemm_cpu_impl('T', 'C', A.extent(0), B.extent(1), A.extent(1), pointerConverter(&alpha),
		  pointerConverter(A.data()), A.extent(0), pointerConverter(B.data()), B.extent(0),
		  pointerConverter(&beta), pointerConverter(C.data()), C.extent(0));
  }
};  

#ifdef KOKKOS_ENABLE_CUDA

template<typename valueType, typename layoutType>
class gpuLinalgHelper {
public:
  using viewType = Kokkos::View<valueType**, layoutType>;
  using arrType = Kokkos::View<valueType*>;
private:
  double* pointerConverter(double* d) {  return d; }
  float* pointerConverter(float* d) {  return d; }
  cuFloatComplex* pointerConverter(cuFloatComplex* d) { return d; }
  cuFloatComplex* pointerConverter(std::complex<float>* d) { return (cuFloatComplex*)d; }
  cuFloatComplex* pointerConverter(Kokkos::complex<float>* d) { return (cuFloatComplex*)d; }
  cuDoubleComplex* pointerConverter(cuDoubleComplex* d) { return d; }
  cuDoubleComplex* pointerConverter(std::complex<double>* d) { return (cuDoubleComplex*)d; }
  cuDoubleComplex* pointerConverter(Kokkos::complex<double>* d) { return (cuDoubleComplex*)d; }

  double** pointerConverter(double** d) {  return d; }
  float** pointerConverter(float** d) {  return d; }
  cuFloatComplex** pointerConverter(cuFloatComplex** d) { return d; }
  cuFloatComplex** pointerConverter(std::complex<float>** d) { return (cuFloatComplex**)d; }
  cuFloatComplex** pointerConverter(Kokkos::complex<float>** d) { return (cuFloatComplex**)d; }
  cuDoubleComplex** pointerConverter(cuDoubleComplex** d) { return d; }
  cuDoubleComplex** pointerConverter(std::complex<double>** d) { return (cuDoubleComplex**)d; }
  cuDoubleComplex** pointerConverter(Kokkos::complex<double>** d) { return (cuDoubleComplex**)d; }
private:
  cublasHandle_t cublas_handle;
  valueType** devPtrPtr;
  valueType** devOutPtrPtr;
  int status;
  Kokkos::View<int*> piv;
  Kokkos::View<int*> info;
  viewType outView;

public:
  gpuLinalgHelper() : piv("piv", 1), info("info", 1) {
    //checkTemplateParams<valueType, Kokkos::LayoutLeft, Kokkos::CudaSpace>();
    status = -1;

    cublasCreate(&cublas_handle);
    cudaMalloc<valueType*>(&devPtrPtr, sizeof(valueType*));
    cudaMalloc<valueType*>(&devOutPtrPtr, sizeof(valueType*));
  }

  ~gpuLinalgHelper() {
    cublasDestroy(cublas_handle);
    cudaFree(devPtrPtr);
    cudaFree(devOutPtrPtr);
  }

  Kokkos::View<int*>::HostMirror extractPivot() {
    Kokkos::View<int*>::HostMirror piv_mirror = Kokkos::create_mirror_view(piv);
    Kokkos::deep_copy(piv_mirror, piv);
    return piv_mirror;
  }

  void getrf(viewType view) {
    int ext = view.extent(0);
    if(piv.extent(0) != ext) {
      Kokkos::resize(piv, ext);
    }
    valueType* tmp = view.data();
    valueType** temp_host_ptr = &tmp; // taking the address on the host
    cudaMemcpy(devPtrPtr,temp_host_ptr,sizeof(temp_host_ptr),cudaMemcpyHostToDevice); // copy the address to dev_ptr
    getrf_gpu_impl(ext,pointerConverter(devPtrPtr),piv.data(),info.data(),cublas_handle);
  }
  void getri(viewType view) {
    int ext = view.extent(0);
    if(piv.extent(0) != ext) {
      Kokkos::resize(piv, ext);
    }
    Kokkos::Profiling::pushRegion("getri_pointer_setup");
    valueType* tmp = view.data();
    valueType** temp_host_ptr = &tmp; // taking the address on the host
    cudaMemcpy(devPtrPtr,temp_host_ptr,sizeof(temp_host_ptr),cudaMemcpyHostToDevice); // copy the address to dev_ptr

    if (outView.extent(0) != ext || outView.extent(1) != ext) {
      Kokkos::resize(outView, ext, ext);
    }
    tmp = outView.data();
    temp_host_ptr = &tmp; // taking the address on the host
    cudaMemcpy(devOutPtrPtr,temp_host_ptr,sizeof(temp_host_ptr),cudaMemcpyHostToDevice); // copy the address to dev_ptr
    Kokkos::Profiling::popRegion();

    Kokkos::Profiling::pushRegion("getri::cublas");
    getri_gpu_impl(ext,pointerConverter(devPtrPtr),pointerConverter(devOutPtrPtr),piv.data(),info.data(),cublas_handle);
    Kokkos::Profiling::popRegion();
    Kokkos::deep_copy(view, outView);

    
  }
  void invertMatrix(viewType view) {
    getrf(view);
    getri(view);
  }
    void gemvTrans(viewType A, arrType x, arrType y, valueType alpha, valueType beta) {
    
    gemv_gpu_impl(cublas_handle,CUBLAS_OP_T, A.extent(0), A.extent(1), pointerConverter(&alpha), 
		  pointerConverter(A.data()), A.extent(0), pointerConverter(x.data()),
		  1, pointerConverter(&beta), pointerConverter(y.data()), 1);
  }
  void gemvConj(viewType A, arrType x, arrType y, valueType alpha, valueType beta) {
    gemv_gpu_impl(cublas_handle,CUBLAS_OP_C, A.extent(0), A.extent(1), pointerConverter(&alpha), 
		  pointerConverter(A.data()), A.extent(0), pointerConverter(x.data()),
		  1, pointerConverter(&beta), pointerConverter(y.data()), 1);
  }
  void gemvNorm(viewType A, arrType x, arrType y, valueType alpha, valueType beta) {
    gemv_gpu_impl(cublas_handle,CUBLAS_OP_N, A.extent(0), A.extent(1), pointerConverter(&alpha), 
		  pointerConverter(A.data()), A.extent(0), pointerConverter(x.data()),
		  1, pointerConverter(&beta), pointerConverter(y.data()), 1); 
  }
  void ger(viewType A, arrType x, arrType y, valueType alpha) {
    ger_gpu_impl(cublas_handle, A.extent(0), A.extent(1), pointerConverter(&alpha), pointerConverter(x.data()), 
		 1, pointerConverter(y.data()), 1, pointerConverter(A.data()), A.extent(0));
  }
    void gemmNN(viewType A, viewType B, viewType C, valueType alpha, valueType beta) {
    gemm_gpu_impl(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, A.extent(0), B.extent(1), A.extent(1), pointerConverter(&alpha),
		  pointerConverter(A.data()), A.extent(0), pointerConverter(B.data()), B.extent(0),
		  pointerConverter(&beta), pointerConverter(C.data()), C.extent(0));
  }
  void gemmNT(viewType A, viewType B, viewType C, valueType alpha, valueType beta) {
    gemm_gpu_impl(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, A.extent(0), B.extent(1), A.extent(1), pointerConverter(&alpha),
		  pointerConverter(A.data()), A.extent(0), pointerConverter(B.data()), B.extent(0),
		  pointerConverter(&beta), pointerConverter(C.data()), C.extent(0));
  }
  void gemmNC(viewType A, viewType B, viewType C, valueType alpha, valueType beta) {
    gemm_gpu_impl(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_C, A.extent(0), B.extent(1), A.extent(1), pointerConverter(&alpha),
		  pointerConverter(A.data()), A.extent(0), pointerConverter(B.data()), B.extent(0),
		  pointerConverter(&beta), pointerConverter(C.data()), C.extent(0));
  }
  void gemmTN(viewType A, viewType B, viewType C, valueType alpha, valueType beta) {
    gemm_gpu_impl(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, A.extent(0), B.extent(1), A.extent(1), pointerConverter(&alpha),
		  pointerConverter(A.data()), A.extent(0), pointerConverter(B.data()), B.extent(0),
		  pointerConverter(&beta), pointerConverter(C.data()), C.extent(0));
  }
  void gemmTT(viewType A, viewType B, viewType C, valueType alpha, valueType beta) {
    gemm_gpu_impl(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, A.extent(0), B.extent(1), A.extent(1), pointerConverter(&alpha),
		  pointerConverter(A.data()), A.extent(0), pointerConverter(B.data()), B.extent(0),
		  pointerConverter(&beta), pointerConverter(C.data()), C.extent(0));
  }
  void gemmTC(viewType A, viewType B, viewType C, valueType alpha, valueType beta) {
    gemm_gpu_impl(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_C, A.extent(0), B.extent(1), A.extent(1), pointerConverter(&alpha),
		  pointerConverter(A.data()), A.extent(0), pointerConverter(B.data()), B.extent(0),
		  pointerConverter(&beta), pointerConverter(C.data()), C.extent(0));
  }
  void gemmCN(viewType A, viewType B, viewType C, valueType alpha, valueType beta) {
    gemm_gpu_impl(cublas_handle, CUBLAS_OP_C, CUBLAS_OP_N, A.extent(0), B.extent(1), A.extent(1), pointerConverter(&alpha),
		  pointerConverter(A.data()), A.extent(0), pointerConverter(B.data()), B.extent(0),
		  pointerConverter(&beta), pointerConverter(C.data()), C.extent(0));
  }
  void gemmCT(viewType A, viewType B, viewType C, valueType alpha, valueType beta) {
    gemm_gpu_impl(cublas_handle, CUBLAS_OP_C, CUBLAS_OP_T, A.extent(0), B.extent(1), A.extent(1), pointerConverter(&alpha),
		  pointerConverter(A.data()), A.extent(0), pointerConverter(B.data()), B.extent(0),
		  pointerConverter(&beta), pointerConverter(C.data()), C.extent(0));
  }
  void gemmCC(viewType A, viewType B, viewType C, valueType alpha, valueType beta) {
    gemm_gpu_impl(cublas_handle, CUBLAS_OP_C, CUBLAS_OP_C, A.extent(0), B.extent(1), A.extent(1), pointerConverter(&alpha),
		  pointerConverter(A.data()), A.extent(0), pointerConverter(B.data()), B.extent(0),
		  pointerConverter(&beta), pointerConverter(C.data()), C.extent(0));
  }

};
  
template<typename valueType, typename layoutType, typename memorySpace>
class linalgHelper<valueType,layoutType,Kokkos::CudaSpace> : public gpuLinalgHelper<valueType,layoutType> {
};

template<typename valueType, typename layoutType, typename memorySpace>
class linalgHelper<valueType,layoutType,Kokkos::CudaUVMSpace> : public gpuLinalgHelper<valueType,layoutType> {
};

#endif


// does matrix operation a * transpose(b) and then checks whether this is different from the identity
template<class viewType1, class viewType2, class linAlgHelper>
void checkIdentity(viewType1 a, viewType2 b, const std::string& tag, linAlgHelper& lah) {
  using vt = typename viewType1::value_type;
  vt error = 0.0;
  vt cone = 1.0;
  viewType1 result("result", a.extent(0), a.extent(1));
  auto result_h = Kokkos::create_mirror_view(result);
  lah.gemmNT(a, b, result, 1.0, 0.0);

  Kokkos::deep_copy(result_h, result);

  for (int i = 0; i < a.extent(0); i++) {
    for (int j = 0; j < b.extent(1); j++) {
      error += (i == j) ? std::abs(result_h(i,j) - cone) : std::abs(result_h(i,j));
    }
  }
  std::cout << tag << " difference from identity (average per element) = " << error / a.extent(0) / a.extent(1) << std::endl;
}

template<class viewType1, class viewType2>
void checkDiff(viewType1 a, viewType2 b, const std::string& tag) {
  using vt = typename viewType1::value_type;
  vt error = 0.0;
  const int dim0 = a.extent(0);
  const int dim1 = a.extent(1);
  Kokkos::parallel_reduce(dim0*dim1, KOKKOS_LAMBDA (int ii, vt& update) {
      int i = ii / dim0;
      int j = ii % dim0;
      update += abs(a(i,j) -b(i,j));
  }, error);
  Kokkos::fence();
  std::cout << tag << " difference between matrices (average per element) = " << error / dim0 / dim1 << std::endl;
}


template<class ViewType, class LinAlgHelperType, typename value_type>
value_type InvertWithLog(ViewType view, LinAlgHelperType& lah, value_type& phase) {
  value_type logdet(0.0);
  lah.getrf(view);
  auto piv = lah.extractPivot();
  int sign_det = 1;
  for (int i = 0; i < view.extent(0); i++) {
    sign_det *= (piv(i) == i+1) ? 1 : -1;
    sign_det *= (view(i,i) > 0) ? 1 : -1;
    logdet += std::log(std::abs(view(i,i)));
  }
  lah.getri(view);
  phase = (sign_det > 0) ? 0.0 : M_PI;
  return logdet;
}

template<class ViewType, class ArrayViewType, class LinAlgHelperType, typename value_type>
void updateRow(ViewType pinv, ArrayViewType tv, int rowchanged, value_type c_ratio_in, LinAlgHelperType& lah) {
  constexpr value_type cone(1.0);
  constexpr value_type czero(0.0);
  ArrayViewType temp("temp", tv.extent(0));
  ArrayViewType rcopy("rcopy", tv.extent(0));
  value_type c_ratio = cone / c_ratio_in;
  lah.gemvTrans(pinv, tv, temp, c_ratio, czero);

  // hard work to modify one element of temp on the device
  auto devElem = subview(temp, rowchanged);
  auto devElem_mirror = Kokkos::create_mirror_view(devElem);
  devElem_mirror(0) = cone - c_ratio;
  Kokkos::deep_copy(devElem, devElem_mirror);

  // now extract the proper part of pinv into rcopy
  // in previous version this looked like: std::copy_n(pinv + m * rowchanged, m, rcopy);
  // a little concerned about getting the ordering wrong
  Kokkos::parallel_for(tv.extent(0), KOKKOS_LAMBDA(int i) {
      rcopy(i) = pinv(rowchanged,i);
  });
  Kokkos::fence();
      
  // now do ger
  lah.ger(pinv, rcopy, temp, -cone);
}

			 

      

struct DiracDeterminant : public WaveFunctionComponent
{
  DiracDeterminant(int nels, const RandomGenerator<RealType>& RNG, int First = 0) 
    : FirstIndex(First), myRandom(RNG)
  {
    psiMinv = DoubleMatType("psiMinv",nels,nels);
    psiM = DoubleMatType("psiM",nels,nels);
    psiMsave = DoubleMatType("psiMsave",nels,nels);
    psiV = Kokkos::View<RealType*>("psiV",nels);

    psiMinv_host = Kokkos::create_mirror_view(psiMinv);
    psiMsave_host = Kokkos::create_mirror_view(psiMsave);
    psiM_host = Kokkos::create_mirror_view(psiM);
    psiV_host = Kokkos::create_mirror_view(psiV);

    

    // basically we are generating uniform random number for
    // each entry of psiMsave in the interval [-0.5, 0.5]
    constexpr double shift(0.5);

    // change this to match data ordering of DeterminantRef
    // recall that psiMsave has as its fast index the leftmost index
    // however int the c style matrix in DeterminantRef 
    for (int i = 0; i < nels; i++) {
      for (int j = 0; j < nels; j++) {
	psiMsave_host(i,j) = myRandom.rand()-shift;
      }
    }
    Kokkos::deep_copy(psiMsave, psiMsave_host);
     
    RealType phase;
    
    for (int i = 0; i < nels; i++) {
      for (int j = 0; j < nels; j++) {
	psiM_host(i,j) = psiMsave_host(j,i);
      }
    }
    Kokkos::deep_copy(psiM, psiM_host);

    LogValue = InvertWithLog(psiM, lah, phase);
    elementWiseCopy(psiMinv, psiM);
  }
  void checkMatrix()
  {
    MatType psiMRealType("psiM_RealType", psiM.extent(0), psiM.extent(0));
    elementWiseCopy(psiMRealType, psiM);
    checkIdentity(psiMsave, psiMRealType, "Psi_0 * psiM(T)", lah);
    checkIdentity(psiMsave, psiMinv, "Psi_0 * psiMinv(T)", lah);
    checkDiff(psiMRealType, psiMinv, "psiM - psiMinv(T)");
  }
  RealType evaluateLog(ParticleSet& P,
		       ParticleSet::ParticleGradient_t& G,
		       ParticleSet::ParticleLaplacian_t& L)
  {
    recompute();
    return 0.0;
  }

  GradType evalGrad(ParticleSet& P, int iat) { return GradType(); }
  ValueType ratioGrad(ParticleSet& P, int iat, GradType& grad) { return ratio(P, iat); }
  void evaluateGL(ParticleSet& P,
                  ParticleSet::ParticleGradient_t& G,
                  ParticleSet::ParticleLaplacian_t& L,
                  bool fromscratch = false)
  {}
  inline void recompute()
  {
    //elementWiseCopy(psiM, psiMsave); // needs to be transposed!
    elementWiseCopyTrans(psiM, psiMsave); // needs to be transposed!
    lah.invertMatrix(psiM);
    elementWiseCopy(psiMinv, psiM);
  }
  inline ValueType ratio(ParticleSet& P, int iel)
  {
    const int nels = psiV.extent(0);
    constexpr double shift(0.5);
    //constexpr double czero(0);
    for (int j = 0; j < nels; ++j) {
      psiV_host(j) = myRandom() - shift;
    }
    Kokkos::deep_copy(psiV, psiV_host);
    // in main line previous version this looked like:
    // curRatio = inner_product_n(psiV.data(), psiMinv[iel - FirstIndex], nels, czero);
    // same issues with indexing
    const int FirstIndex_ = FirstIndex;
    const int iel_=iel;
    double curRatio_=0;
    Kokkos::View<RealType*> psiV_=psiV;
    DoubleMatType psiMinv_=psiMinv; 
    Kokkos::parallel_reduce( nels, KOKKOS_LAMBDA (int i, ValueType& update) {
	update += psiV_(i) * psiMinv_(iel_-FirstIndex_,i);
    }, curRatio_);
    Kokkos::fence();
    curRatio=curRatio_;
    return curRatio;
  }
  inline void acceptMove(ParticleSet& P, int iel) {
    const int nels = psiV.extent(0);
    updateRow(psiMinv, psiV, iel, curRatio, lah);
    // in main line previous version this looked like:
    //std::copy_n(psiV.data(), nels, psiMsave[iel - FirstIndex]);
    // note 1: copy_n copies data from psiV to psiMsave
    //
    // note 2: the single argument call to psiMsave[] returned a pointer to
    // the iel-FirstIndex ROW of the underlying data structure, so
    // the operation was like (psiMsave.data() + (iel-FirstIndex)*D2)
    // note that then to iterate through the data it was going sequentially
    const int FirstIndex_ = FirstIndex;
    Kokkos::View<RealType*> psiV_=psiV;
    const int iel_=iel;
    DoubleMatType psiMsave_=psiMsave; 
    Kokkos::parallel_for( nels, KOKKOS_LAMBDA (int i) {
    	psiMsave_(iel_-FirstIndex_, i) = psiV_(i);
      });
    Kokkos::fence();
  }

  // accessor functions for checking
  inline double operator()(int i) const {
    Kokkos::deep_copy(psiMinv, psiMinv_host);
    int x = i / psiMinv_host.extent(0);
    int y = i % psiMinv_host.extent(0);
    auto dev_subview = subview(psiMinv, x, y);
    auto dev_subview_host = Kokkos::create_mirror_view(dev_subview);
    Kokkos::deep_copy(dev_subview_host, dev_subview);
    return dev_subview_host(0,0);
  }
  inline int size() const { return psiMinv.extent(0)*psiMinv.extent(1); }

private:
  /// log|det|
  double LogValue;
  /// current ratio
  double curRatio;
  /// initial particle index
  const int FirstIndex;
  /// matrix type and mirror type
  //using MatType = Kokkos::View<RealType**, Kokkos::LayoutLeft>;
  using MatType = Kokkos::View<RealType**, Kokkos::LayoutRight>;
  using MatMirrorType = MatType::HostMirror;
  //using DoubleMatType = Kokkos::View<double**, Kokkos::LayoutLeft>;
  using DoubleMatType = Kokkos::View<double**, Kokkos::LayoutRight>;
  using DoubleMatMirrorType = DoubleMatType::HostMirror;
  /// inverse matrix to be updated and host mirror (kept in double regardless of RealType)
  MatType psiMinv;
  MatMirrorType psiMinv_host;
  /// storage for the row update and host mirror
  Kokkos::View<RealType*> psiV;
  Kokkos::View<RealType*>::HostMirror psiV_host;
  /// internal storage to perform inversion correctly and host mirror
  DoubleMatType psiM;
  DoubleMatMirrorType psiM_host;
  /// temporary workspace for inversion and host mirror
  MatType psiMsave;
  MatMirrorType psiMsave_host;
  /// random number generator for testing
  RandomGenerator<RealType> myRandom;
  /// Helper class to handle linear algebra
  /// Holds for instance space for pivots and workspace
  linalgHelper<MatType::value_type, MatType::array_layout, MatType::memory_space> lah;
};
} // namespace qmcplusplus







#endif
