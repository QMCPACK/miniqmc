#include "QMCWaveFunctions/DeterminantDeviceImpKOKKOS.h"


void getrf_cpu_impl(const int& n, const int& m, float* a, const int& n0, int* piv, int& st)
{
  sgetrf(n, m, a, n0, piv, st);
}
void getrf_cpu_impl(const int& n, const int& m, double* a, const int& n0, int* piv, int& st)
{
  dgetrf(n, m, a, n0, piv, st);
}
void getrf_cpu_impl(const int& n, const int& m, std::complex<float>* a, const int& n0, int* piv, int& st)
{
  cgetrf(n, m, a, n0, piv, st);
}
void getrf_cpu_impl(const int& n, const int& m, std::complex<double>* a, const int& n0, int* piv, int& st)
{
  zgetrf(n, m, a, n0, piv, st);
}

void getri_cpu_impl(const int& n, float* a, const int& n0, int* piv, float* work, const int& lwork, int& st)
{
  sgetri(n, a, n0, piv, work, lwork, st);
}
void getri_cpu_impl(const int& n,
                    double* a,
                    const int& n0,
                    int* piv,
                    double* work,
                    const int& lwork,
                    int& st)
{
  dgetri(n, a, n0, piv, work, lwork, st);
}
void getri_cpu_impl(const int& n,
                    std::complex<float>* a,
                    const int& n0,
                    int* piv,
                    std::complex<float>* work,
                    const int& lwork,
                    int& st)
{
  cgetri(n, a, n0, piv, work, lwork, st);
}
void getri_cpu_impl(const int& n,
                    std::complex<double>* a,
                    const int& n0,
                    int* piv,
                    std::complex<double>* work,
                    const int& lwork,
                    int& st)
{
  zgetri(n, a, n0, piv, work, lwork, st);
}

void gemv_cpu_impl(const char& trans,
                   const int& nr,
                   const int& nc,
                   const float& alpha,
                   const float* amat,
                   const int& lda,
                   const float* bv,
                   const int& incx,
                   const float& beta,
                   float* cv,
                   const int& incy)
{
  sgemv(trans, nr, nc, alpha, amat, lda, bv, incx, beta, cv, incy);
}
void gemv_cpu_impl(const char& trans,
                   const int& nr,
                   const int& nc,
                   const double& alpha,
                   const double* amat,
                   const int& lda,
                   const double* bv,
                   const int& incx,
                   const double& beta,
                   double* cv,
                   const int& incy)
{
  Kokkos::Profiling::pushRegion("gemv_cpu_impl::dgemv");
  dgemv(trans, nr, nc, alpha, amat, lda, bv, incx, beta, cv, incy);
  Kokkos::Profiling::popRegion();
}
void gemv_cpu_impl(const char& trans,
                   const int& nr,
                   const int& nc,
                   const std::complex<float>& alpha,
                   const std::complex<float>* amat,
                   const int& lda,
                   const std::complex<float>* bv,
                   const int& incx,
                   const std::complex<float>& beta,
                   std::complex<float>* cv,
                   const int& incy)
{
  cgemv(trans, nr, nc, alpha, amat, lda, bv, incx, beta, cv, incy);
}
void gemv_cpu_impl(const char& trans,
                   const int& nr,
                   const int& nc,
                   const std::complex<double>& alpha,
                   const std::complex<double>* amat,
                   const int& lda,
                   const std::complex<double>* bv,
                   const int& incx,
                   const std::complex<double>& beta,
                   std::complex<double>* cv,
                   const int& incy)
{
  zgemv(trans, nr, nc, alpha, amat, lda, bv, incx, beta, cv, incy);
}

void ger_cpu_impl(const int& m,
                  const int& n,
                  const float& alpha,
                  const float* x,
                  const int& incx,
                  const float* y,
                  const int& incy,
                  float* a,
                  const int& lda)
{
  sger(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
}
void ger_cpu_impl(const int& m,
                  const int& n,
                  const double& alpha,
                  const double* x,
                  const int& incx,
                  const double* y,
                  const int& incy,
                  double* a,
                  const int& lda)
{
  Kokkos::Profiling::pushRegion("ger_cpu_impl");
  dger(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
  Kokkos::Profiling::popRegion();
}
void ger_cpu_impl(const int& m,
                  const int& n,
                  const std::complex<float>& alpha,
                  const std::complex<float>* x,
                  const int& incx,
                  const std::complex<float>* y,
                  const int& incy,
                  std::complex<float>* a,
                  const int& lda)
{
  cgeru(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
}
void ger_cpu_impl(const int& m,
                  const int& n,
                  const std::complex<double>& alpha,
                  const std::complex<double>* x,
                  const int& incx,
                  const std::complex<double>* y,
                  const int& incy,
                  std::complex<double>* a,
                  const int& lda)
{
  zgeru(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
}
void gemm_cpu_impl(const char& transa,
                   const char& transb,
                   const int& rowsa,
                   const int& columnsb,
                   const int& columnsa,
                   const float* alpha,
                   const float* a,
                   const int& lda,
                   const float* b,
                   const int& ldb,
                   const float* beta,
                   float* c,
                   const int& ldc)
{
  sgemm(transa, transb, rowsa, columnsb, columnsa, *alpha, a, lda, b, ldb, *beta, c, ldc);
}
void gemm_cpu_impl(const char& transa,
                   const char& transb,
                   const int& rowsa,
                   const int& columnsb,
                   const int& columnsa,
                   const double* alpha,
                   const double* a,
                   const int& lda,
                   const double* b,
                   const int& ldb,
                   const double* beta,
                   double* c,
                   const int& ldc)
{
  dgemm(transa, transb, rowsa, columnsb, columnsa, *alpha, a, lda, b, ldb, *beta, c, ldc);
}
void gemm_cpu_impl(const char& transa,
                   const char& transb,
                   const int& rowsa,
                   const int& columnsb,
                   const int& columnsa,
                   const std::complex<float>* alpha,
                   const std::complex<float>* a,
                   const int& lda,
                   const std::complex<float>* b,
                   const int& ldb,
                   const std::complex<float>* beta,
                   std::complex<float>* c,
                   const int& ldc)
{
  cgemm(transa, transb, rowsa, columnsb, columnsa, *alpha, a, lda, b, ldb, *beta, c, ldc);
}
void gemm_cpu_impl(const char& transa,
                   const char& transb,
                   const int& rowsa,
                   const int& columnsb,
                   const int& columnsa,
                   const std::complex<double>* alpha,
                   const std::complex<double>* a,
                   const int& lda,
                   const std::complex<double>* b,
                   const int& ldb,
                   const std::complex<double>* beta,
                   std::complex<double>* c,
                   const int& ldc)
{
  zgemm(transa, transb, rowsa, columnsb, columnsa, *alpha, a, lda, b, ldb, *beta, c, ldc);
}


#ifdef KOKKOS_ENABLE_CUDA
void getrf_gpu_impl(int m,
                    int n,
                    float* a,
                    int lda,
                    float* workspace,
                    int* devIpiv,
                    int* devInfo,
                    cusolverDnHandle_t& handle)
{
  auto st = cusolverDnSgetrf(handle, m, n, a, lda, workspace, devIpiv, devInfo);
}
void getrf_gpu_impl(int m,
                    int n,
                    double* a,
                    int lda,
                    double* workspace,
                    int* devIpiv,
                    int* devInfo,
                    cusolverDnHandle_t& handle)
{
  auto st = cusolverDnDgetrf(handle, m, n, a, lda, workspace, devIpiv, devInfo);
}
void getrf_gpu_impl(int m,
                    int n,
                    cuFloatComplex* a,
                    int lda,
                    cuFloatComplex* workspace,
                    int* devIpiv,
                    int* devInfo,
                    cusolverDnHandle_t& handle)
{
  auto st = cusolverDnCgetrf(handle, m, n, a, lda, workspace, devIpiv, devInfo);
}
void getrf_gpu_impl(int m,
                    int n,
                    cuDoubleComplex* a,
                    int lda,
                    cuDoubleComplex* workspace,
                    int* devIpiv,
                    int* devInfo,
                    cusolverDnHandle_t& handle)
{
  auto st = cusolverDnZgetrf(handle, m, n, a, lda, workspace, devIpiv, devInfo);
}

int getrf_gpu_buffer_size(int m, int n, float* a, int lda, cusolverDnHandle_t& handle)
{
  int lwork;
  auto st = cusolverDnSgetrf_bufferSize(handle, m, n, a, lda, &lwork);
  return lwork;
}
int getrf_gpu_buffer_size(int m, int n, double* a, int lda, cusolverDnHandle_t& handle)
{
  int lwork;
  auto st = cusolverDnDgetrf_bufferSize(handle, m, n, a, lda, &lwork);
  return lwork;
}
int getrf_gpu_buffer_size(int m, int n, cuFloatComplex* a, int lda, cusolverDnHandle_t& handle)
{
  int lwork;
  auto st = cusolverDnCgetrf_bufferSize(handle, m, n, a, lda, &lwork);
  return lwork;
}
int getrf_gpu_buffer_size(int m, int n, cuDoubleComplex* a, int lda, cusolverDnHandle_t& handle)
{
  int lwork;
  auto st = cusolverDnZgetrf_bufferSize(handle, m, n, a, lda, &lwork);
  return lwork;
}

// note on input b should be an n by n identity matrix
// also, all strides should be 1
void getri_gpu_impl(const int n, float* a, int* piv, float* b, int* info, cusolverDnHandle_t& handle)
{
  auto st = cusolverDnSgetrs(handle, CUBLAS_OP_N, n, n, a, n, piv, b, n, info);
}
void getri_gpu_impl(const int n, double* a, int* piv, double* b, int* info, cusolverDnHandle_t& handle)
{
  auto st = cusolverDnDgetrs(handle, CUBLAS_OP_N, n, n, a, n, piv, b, n, info);
}
void getri_gpu_impl(const int n,
                    cuFloatComplex* a,
                    int* piv,
                    cuFloatComplex* b,
                    int* info,
                    cusolverDnHandle_t& handle)
{
  auto st = cusolverDnCgetrs(handle, CUBLAS_OP_N, n, n, a, n, piv, b, n, info);
}
void getri_gpu_impl(const int n,
                    cuDoubleComplex* a,
                    int* piv,
                    cuDoubleComplex* b,
                    int* info,
                    cusolverDnHandle_t& handle)
{
  auto st = cusolverDnZgetrs(handle, CUBLAS_OP_N, n, n, a, n, piv, b, n, info);
}


void gemv_gpu_impl(cublasHandle_t& handle,
                   cublasOperation_t trans,
                   const int& nr,
                   const int& nc,
                   const float* alpha,
                   const float* amat,
                   const int& lda,
                   const float* bv,
                   const int& incx,
                   const float* beta,
                   float* cv,
                   const int& incy)
{
  cublasSgemv(handle, trans, nr, nc, alpha, amat, lda, bv, incx, beta, cv, incy);
}
void gemv_gpu_impl(cublasHandle_t& handle,
                   cublasOperation_t trans,
                   const int& nr,
                   const int& nc,
                   const double* alpha,
                   const double* amat,
                   const int& lda,
                   const double* bv,
                   const int& incx,
                   const double* beta,
                   double* cv,
                   const int& incy)
{
  cublasDgemv(handle, trans, nr, nc, alpha, amat, lda, bv, incx, beta, cv, incy);
}
void gemv_gpu_impl(cublasHandle_t& handle,
                   cublasOperation_t trans,
                   const int& nr,
                   const int& nc,
                   const cuFloatComplex* alpha,
                   const cuFloatComplex* amat,
                   const int& lda,
                   const cuFloatComplex* bv,
                   const int& incx,
                   const cuFloatComplex* beta,
                   cuFloatComplex* cv,
                   const int& incy)
{
  cublasCgemv(handle, trans, nr, nc, alpha, amat, lda, bv, incx, beta, cv, incy);
}
void gemv_gpu_impl(cublasHandle_t& handle,
                   cublasOperation_t trans,
                   const int& nr,
                   const int& nc,
                   const cuDoubleComplex* alpha,
                   const cuDoubleComplex* amat,
                   const int& lda,
                   const cuDoubleComplex* bv,
                   const int& incx,
                   const cuDoubleComplex* beta,
                   cuDoubleComplex* cv,
                   const int& incy)
{
  cublasZgemv(handle, trans, nr, nc, alpha, amat, lda, bv, incx, beta, cv, incy);
}

void ger_gpu_impl(cublasHandle_t& handle,
                  const int& m,
                  const int& n,
                  const float* alpha,
                  const float* x,
                  const int& incx,
                  const float* y,
                  const int& incy,
                  float* a,
                  const int& lda)
{
  cublasSger(handle, m, n, alpha, x, incx, y, incy, a, lda);
}
void ger_gpu_impl(cublasHandle_t& handle,
                  const int& m,
                  const int& n,
                  const double* alpha,
                  const double* x,
                  const int& incx,
                  const double* y,
                  const int& incy,
                  double* a,
                  const int& lda)
{
  cublasDger(handle, m, n, alpha, x, incx, y, incy, a, lda);
}
void ger_gpu_impl(cublasHandle_t& handle,
                  const int& m,
                  const int& n,
                  const cuFloatComplex* alpha,
                  const cuFloatComplex* x,
                  const int& incx,
                  const cuFloatComplex* y,
                  const int& incy,
                  cuFloatComplex* a,
                  const int& lda)
{
  cublasCgeru(handle, m, n, alpha, x, incx, y, incy, a, lda);
}
void ger_gpu_impl(cublasHandle_t& handle,
                  const int& m,
                  const int& n,
                  const cuDoubleComplex* alpha,
                  const cuDoubleComplex* x,
                  const int& incx,
                  const cuDoubleComplex* y,
                  const int& incy,
                  cuDoubleComplex* a,
                  const int& lda)
{
  cublasZgeru(handle, m, n, alpha, x, incx, y, incy, a, lda);
}
void gemm_gpu_impl(cublasHandle_t& handle,
                   cublasOperation_t transa,
                   cublasOperation_t transb,
                   const int& rowsa,
                   const int& columnsb,
                   const int& columnsa,
                   const float* alpha,
                   const float* a,
                   const int& lda,
                   const float* b,
                   const int& ldb,
                   const float* beta,
                   float* c,
                   const int& ldc)
{
  cublasSgemm(handle, transa, transb, rowsa, columnsb, columnsa, alpha, a, lda, b, ldb, beta, c, ldc);
}
void gemm_gpu_impl(cublasHandle_t& handle,
                   cublasOperation_t transa,
                   cublasOperation_t transb,
                   const int& rowsa,
                   const int& columnsb,
                   const int& columnsa,
                   const double* alpha,
                   const double* a,
                   const int& lda,
                   const double* b,
                   const int& ldb,
                   const double* beta,
                   double* c,
                   const int& ldc)
{
  cublasDgemm(handle, transa, transb, rowsa, columnsb, columnsa, alpha, a, lda, b, ldb, beta, c, ldc);
}
void gemm_gpu_impl(cublasHandle_t& handle,
                   cublasOperation_t transa,
                   cublasOperation_t transb,
                   const int& rowsa,
                   const int& columnsb,
                   const int& columnsa,
                   const cuFloatComplex* alpha,
                   const cuFloatComplex* a,
                   const int& lda,
                   const cuFloatComplex* b,
                   const int& ldb,
                   const cuFloatComplex* beta,
                   cuFloatComplex* c,
                   const int& ldc)
{
  cublasCgemm(handle, transa, transb, rowsa, columnsb, columnsa, alpha, a, lda, b, ldb, beta, c, ldc);
}
void gemm_gpu_impl(cublasHandle_t& handle,
                   cublasOperation_t transa,
                   cublasOperation_t transb,
                   const int& rowsa,
                   const int& columnsb,
                   const int& columnsa,
                   const cuDoubleComplex* alpha,
                   const cuDoubleComplex* a,
                   const int& lda,
                   const cuDoubleComplex* b,
                   const int& ldb,
                   const cuDoubleComplex* beta,
                   cuDoubleComplex* c,
                   const int& ldc)
{
  cublasZgemm(handle, transa, transb, rowsa, columnsb, columnsa, alpha, a, lda, b, ldb, beta, c, ldc);
}

namespace qmcplusplus
{
  template class DeterminantDeviceImp<Devices::KOKKOS>;
}

#endif
