////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2017 QMCPACK developers.
//
// File developed by: L. Shulenburger
//
// File created by: L. Shulenburger
////////////////////////////////////////////////////////////////////////////////
// -*- C++ -*-

/**
 * @file LinAlgKokkos.h
 * @brief utility routines to do linear algebra with Kokkos
 */

#ifndef QMCPLUSPLUS_LINALGKOKKOS_H
#define QMCPLUSPLUS_LINALGKOKKOS_H
#include <Kokkos_Core.hpp>
#include <impl/Kokkos_Timer.hpp>

#ifdef KOKKOS_ENABLE_CUDA
#include "cublas_v2.h"
#include "cusolverDn.h"
#endif
#include "Numerics/KokkosViewHelpers.h"

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

namespace qmcplusplus
{

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
  Kokkos::Profiling::pushRegion("gemv_cpu_impl::dgemv");
  dgemv(trans, nr, nc, alpha, amat, lda, bv, incx, beta, cv, incy);
  Kokkos::Profiling::popRegion();
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
  Kokkos::Profiling::pushRegion("ger_cpu_impl");
  dger(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
  Kokkos::Profiling::popRegion();
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
void getrf_gpu_impl(int m, int n, float* a, int lda, float* workspace, int* devIpiv, int* devInfo, cusolverDnHandle_t& handle) {
  auto st = cusolverDnSgetrf(handle, m, n, a, lda, workspace, devIpiv, devInfo);
}
void getrf_gpu_impl(int m, int n, double* a, int lda, double* workspace, int* devIpiv, int* devInfo, cusolverDnHandle_t& handle) {
  auto st = cusolverDnDgetrf(handle, m, n, a, lda, workspace, devIpiv, devInfo);
}
void getrf_gpu_impl(int m, int n, cuFloatComplex* a, int lda, cuFloatComplex* workspace, int* devIpiv, int* devInfo, cusolverDnHandle_t& handle) {
  auto st = cusolverDnCgetrf(handle, m, n, a, lda, workspace, devIpiv, devInfo);
}
void getrf_gpu_impl(int m, int n, cuDoubleComplex* a, int lda, cuDoubleComplex* workspace, int* devIpiv, int* devInfo, cusolverDnHandle_t& handle) {
  auto st = cusolverDnZgetrf(handle, m, n, a, lda, workspace, devIpiv, devInfo);
}

int getrf_gpu_buffer_size(int m, int n, float* a, int lda, cusolverDnHandle_t& handle) {
  int lwork;
  auto st = cusolverDnSgetrf_bufferSize(handle, m, n, a, lda, &lwork);
  return lwork;
}
int getrf_gpu_buffer_size(int m, int n, double* a, int lda, cusolverDnHandle_t& handle) {
  int lwork;
  auto st = cusolverDnDgetrf_bufferSize(handle, m, n, a, lda, &lwork);
  return lwork;
}
int getrf_gpu_buffer_size(int m, int n, cuFloatComplex* a, int lda, cusolverDnHandle_t& handle) {
  int lwork;
  auto st = cusolverDnCgetrf_bufferSize(handle, m, n, a, lda, &lwork);
  return lwork;
}
int getrf_gpu_buffer_size(int m, int n, cuDoubleComplex* a, int lda, cusolverDnHandle_t& handle) {
  int lwork;
  auto st = cusolverDnZgetrf_bufferSize(handle, m, n, a, lda, &lwork);
  return lwork;
}

// note on input b should be an n by n identity matrix
// also, all strides should be 1
void getri_gpu_impl(const int n, float* a, int* piv, float* b, int* info, cusolverDnHandle_t& handle) {
  auto st = cusolverDnSgetrs(handle, CUBLAS_OP_N, n, n, a, n, piv, b, n, info);
}
void getri_gpu_impl(const int n, double* a, int* piv, double* b, int* info, cusolverDnHandle_t& handle) {
  auto st = cusolverDnDgetrs(handle, CUBLAS_OP_N, n, n, a, n, piv, b, n, info);
}
void getri_gpu_impl(const int n, cuFloatComplex* a, int* piv, cuFloatComplex* b, int* info, cusolverDnHandle_t& handle) {
  auto st = cusolverDnCgetrs(handle, CUBLAS_OP_N, n, n, a, n, piv, b, n, info);
}
void getri_gpu_impl(const int n, cuDoubleComplex* a, int* piv, cuDoubleComplex* b, int* info, cusolverDnHandle_t& handle) {
  auto st = cusolverDnZgetrs(handle, CUBLAS_OP_N, n, n, a, n, piv, b, n, info);
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
  
  using arrPolicyParallel_t = Kokkos::RangePolicy<>;
  using arrPolicySerial_t = Kokkos::RangePolicy<Kokkos::Serial>;
  int arrPolicySerialThreshold;

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
  linalgHelper(int serialThreshold = 32768) {
    //checkTemplateParams<valueType, arrayLayout, memorySpace>();
    Kokkos::resize(piv, 1);
    Kokkos::resize(work, 1);
    status = -1;
    arrPolicySerialThreshold = serialThreshold;
  }

  typename Kokkos::View<int*, memorySpace>::HostMirror extractPivot() {
    //typename Kokkos::View<int*, memorySpace>::HostMirror piv_mirror = Kokkos::create_mirror_view(piv);
    auto piv_mirror = Kokkos::create_mirror_view(piv);
    Kokkos::deep_copy(piv_mirror, piv);
    return piv_mirror;
  }

  typename Kokkos::View<int*, memorySpace> getPivot() {
    return piv;
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
    Kokkos::Profiling::pushRegion("linAlgHelper::ger_cpu_impl");
    ger_cpu_impl(A.extent_int(0), A.extent_int(1), alpha, pointerConverter(x.data()), 1, pointerConverter(y.data()), 1,
		 pointerConverter(A.data()), A.extent_int(0));
    Kokkos::Profiling::popRegion();
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

  void copyChangedRow(int rowchanged, viewType pinv, arrType rcopy) {
    if (rcopy.extent(0) > arrPolicySerialThreshold) {
      Kokkos::parallel_for(arrPolicyParallel_t(0, rcopy.extent(0)), KOKKOS_LAMBDA(int i) {
	  rcopy(i) = pinv(rowchanged,i);
	});
    } else {
      Kokkos::parallel_for(arrPolicySerial_t(0, rcopy.extent(0)), KOKKOS_LAMBDA(int i) {
	  rcopy(i) = pinv(rowchanged,i);
	});
    }
    Kokkos::fence();
  }
  
  template<typename doubleViewType>
  valueType updateRatio(arrType psiV, doubleViewType psiMinv, int firstIndex) {
    arrType psiV_ = psiV;
    doubleViewType psiMinv_ = psiMinv;
    valueType curRatio_ = 0.0;
    int firstIndex_ = firstIndex;
    if (psiV_.extent(0) > arrPolicySerialThreshold) {
      Kokkos::parallel_reduce(arrPolicyParallel_t(0, psiV_.extent(0)), KOKKOS_LAMBDA(int i, valueType& update) {
	  update += psiV_(i) * psiMinv_(firstIndex_,i);
	}, curRatio_);
    } else {
      Kokkos::parallel_reduce(arrPolicySerial_t(0, psiV_.extent(0)), KOKKOS_LAMBDA(int i, valueType& update) {
	  update += psiV_(i) * psiMinv_(firstIndex_,i);
	}, curRatio_);
    }
    Kokkos::fence();
    return curRatio_;
  }

  template<typename doubleViewType>
  void copyBack(doubleViewType psiMsave, arrType psiV, int firstIndex) {
    doubleViewType psiMsave_ = psiMsave;
    arrType psiV_ = psiV;
    int firstIndex_ = firstIndex;
    if (psiV_.extent(0) > arrPolicySerialThreshold) {
      Kokkos::parallel_for(arrPolicyParallel_t(0, psiV_.extent(0)), KOKKOS_LAMBDA(int i) {
	  psiMsave_(firstIndex_, i) = psiV_(i);
	});
    } else {
      Kokkos::parallel_for(arrPolicySerial_t(0, psiV_.extent(0)), KOKKOS_LAMBDA(int i) {
	  psiMsave_(firstIndex_, i) = psiV_(i);
	});
    }
    Kokkos::fence();
  }
};

#ifdef KOKKOS_ENABLE_CUDA

template<typename valueType, typename layoutType>
class gpuLinalgHelper {
public:
  using viewType = Kokkos::View<valueType**, layoutType>;
  using arrType = Kokkos::View<valueType*>;

  double* pointerConverter(double* d) {  return d; }
  float* pointerConverter(float* d) {  return d; }
  cuFloatComplex* pointerConverter(cuFloatComplex* d) { return d; }
  cuFloatComplex* pointerConverter(std::complex<float>* d) { return (cuFloatComplex*)d; }
  cuFloatComplex* pointerConverter(Kokkos::complex<float>* d) { return (cuFloatComplex*)d; }
  cuDoubleComplex* pointerConverter(cuDoubleComplex* d) { return d; }
  cuDoubleComplex* pointerConverter(std::complex<double>* d) { return (cuDoubleComplex*)d; }
  cuDoubleComplex* pointerConverter(Kokkos::complex<double>* d) { return (cuDoubleComplex*)d; }

private:
  int status;
  Kokkos::View<int*> piv;
  Kokkos::View<int*> info;
  viewType outView;

public:
  cublasHandle_t cublas_handle;
  cusolverDnHandle_t cusolver_handle;

 gpuLinalgHelper() : piv("piv", 1), info("info", 1) {
    //checkTemplateParams<valueType, Kokkos::LayoutLeft, Kokkos::CudaSpace>();
    status = -1;

    cublasCreate(&cublas_handle);
    cusolverDnCreate(&cusolver_handle);
  }

  ~gpuLinalgHelper() {
    cublasDestroy(cublas_handle);
    cusolverDnDestroy(cusolver_handle);
  }

  Kokkos::View<int*>::HostMirror extractPivot() {
    Kokkos::View<int*>::HostMirror piv_mirror = Kokkos::create_mirror_view(piv);
    Kokkos::deep_copy(piv_mirror, piv);
    return piv_mirror;
  }

  Kokkos::View<int*> getPivot() {
    return piv;
  }

  void getrf(viewType view) {
    int m = view.extent(0);
    int n = view.extent(1);
    
    int worksize = getrf_gpu_buffer_size(m, n, pointerConverter(view.data()), m, cusolver_handle);
    arrType workspace("getrf_workspace", worksize);
    
    if(piv.extent(0) != m) {
      Kokkos::resize(piv,m);
    }
    getrf_gpu_impl(m, n, pointerConverter(view.data()), m, pointerConverter(workspace.data()), piv.data(), info.data(), cusolver_handle);
  }

  void getri(viewType view) {
    int n = view.extent(0);
    // make identity matrix for right hand side
    viewType outView("outputView", n, n);
    Kokkos::parallel_for(n, KOKKOS_LAMBDA(int i) {
      outView(i,i) = 1.0;
      });
    getri_gpu_impl(n, pointerConverter(view.data()), piv.data(), pointerConverter(outView.data()), info.data(), cusolver_handle);
    // now copy back to view from outputView
    //Kokkos::deep_copy(view, outView);
    elementWiseCopy(view, outView);
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
  void copyChangedRow(int rowchanged, viewType pinv, arrType rcopy) {
    viewType pinv_ = pinv;
    arrType rcopy_ = rcopy;
    Kokkos::parallel_for(rcopy_.extent(0), KOKKOS_LAMBDA(int i) {
	rcopy_(i) = pinv_(rowchanged,i);
      });
    Kokkos::fence();
  }
  template<typename doubleViewType>
  valueType updateRatio(arrType psiV, doubleViewType psiMinv, int firstIndex) {
    arrType psiV_ = psiV;
    doubleViewType psiMinv_ = psiMinv;
    valueType curRatio_ = 0.0;
    int firstIndex_ = firstIndex;
    Kokkos::parallel_reduce( psiV_.extent_int(0), KOKKOS_LAMBDA (int i, valueType& update) {
	update += psiV_(i) * psiMinv_(firstIndex_,i);
    }, curRatio_);
    return curRatio_;
  }
  template<typename doubleViewType>
  void copyBack(doubleViewType psiMsave, arrType psiV, int firstIndex) {
    doubleViewType psiMsave_ = psiMsave;
    arrType psiV_ = psiV;
    int firstIndex_ = firstIndex;
    Kokkos::parallel_for( psiV_.extent_int(0), KOKKOS_LAMBDA (int i) {
    	psiMsave_(firstIndex_, i) = psiV_(i);
      });
    Kokkos::fence();
  }

};
  
template<typename valueType, typename layoutType>
class linalgHelper<valueType,layoutType,Kokkos::CudaSpace> : public gpuLinalgHelper<valueType,layoutType> {
};

template<typename valueType, typename layoutType>
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


} // end namespace qmcplusplus
#endif
