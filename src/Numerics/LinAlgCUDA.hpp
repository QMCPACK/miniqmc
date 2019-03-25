#ifndef QMCPLUSPLUS_LINALG_CUDA_H
#define QMCPLUSPLUS_LINALG_CUDA_H

#include "cuda.h"
#include "cublas_v2.h"
#include "LinAlgCPU.h"
namespace qmcplusplus
{

class HandleToCublas
{
private:
    cublasHandle_t handle;
  HandleToCublas() { cublasCreate(&handle); }
  ~HandleToCublas() { cublasDestroy(handle); }
public:
  static cublasHandle_t& get()
  {
    static HandleToCublas instance;
    return instance.handle;
  }
};

namespace linalg
{
//inlines below just to avoid problems with inclusion in multiple compilation units for now
    // These should just be explicitly instantiated and linked.
    
template<typename T>
inline void TGemV(int n, int m, T alpha, const T *restrict amat, int lda,
     const T* x, int incx, T beta, T* y, int incy);

template<>
inline void TGemV(int n, int m, double alpha, const double *restrict amat, int lda,
     const double* x, int incx, double beta, double* y, int incy)
{
    cublasSetStream(HandleToCublas::get(),cudaStreamPerThread);
    cublasStatus_t err = cublasDgemv(HandleToCublas::get(),
				     cublasOperation_t::CUBLAS_OP_T,
				     m,
				     n,
				     &alpha,
				     amat,
				     lda,
				     x,
				     incx,
				     &beta,
				     y,
				     incy);
    if(err != cublasStatus_t::CUBLAS_STATUS_SUCCESS)
	throw std::runtime_error("cublasDgemv failed");
}

template <typename T>
inline void Ger(int m, int n, T alpha, const T *x, int incx,
	 const T *y, int incy, T *a, int lda);

template<>
inline void Ger(int m, int n, double alpha, const double *x, int incx,
	 const double *y, int incy, double *a, int lda)
{
    cublasSetStream(HandleToCublas::get(),cudaStreamPerThread);
  cublasStatus_t err = cublasDger(HandleToCublas::get(),
				  m,
				  n,
				  &alpha,
				  x,
				  incx,
				  y,
				  incy,
				  a,
				  lda);

}
    
}
struct LinAlgCUDA : public LinAlgCPU
{
  template<typename T, typename RT>
  inline void
  updateRow(Matrix<T>& psiMinv, aligned_vector<T>& psiV, int m, int lda, int rowchanged, RT c_ratio_in, size_t cuda_buffer_size, void* cuda_buffer)
  {
    constexpr T cone(1);
    constexpr T czero(0);
    T temp[m];
    T rcopy[m];
    T* dev_ptr = static_cast<T*>(cuda_buffer);
    char* buffer = new char[cuda_buffer_size];
    char * temp_buffer = buffer + (psiMinv.size() + psiV.size())* sizeof(T);
    cudaMemcpy(dev_ptr, psiMinv.data(), psiMinv.size() * sizeof(T),cudaMemcpyHostToDevice);
    T* psi_v_dev_ptr = (double*)dev_ptr + psiMinv.size(); 
    cudaMemcpy(psi_v_dev_ptr, psiV.data(), psiV.size() * sizeof(T),cudaMemcpyHostToDevice);
    T* temp_dev_ptr = psi_v_dev_ptr + psiV.size();
    T c_ratio = cone / c_ratio_in;
    linalg::TGemV(m, m, c_ratio, dev_ptr, m, psi_v_dev_ptr, 1, czero, temp_dev_ptr, 1);
    //BLAS::gemv('T', m, m, c_ratio, pinv, m, tv, 1, czero, temp, 1);
    cudaMemcpy((void*)buffer, dev_ptr, cuda_buffer_size, cudaMemcpyDeviceToHost);
    std::memcpy(temp, temp_buffer, m * sizeof(T));
    temp[rowchanged] = cone - c_ratio;
    std::copy_n(psiMinv.data() + m * rowchanged, m, rcopy);
    std::memcpy(temp_buffer, temp, m * sizeof(T));
    std::memcpy(temp_buffer + m * sizeof(T), rcopy, m * sizeof(T));
    cudaMemcpy(temp_dev_ptr, temp_buffer, 2 * m * sizeof(T),cudaMemcpyHostToDevice);
    linalg::Ger(m, m, -cone, temp_dev_ptr+m, 1, temp_dev_ptr, 1, dev_ptr, m);
    cudaMemcpy(psiMinv.data(), dev_ptr, psiMinv.size() * sizeof(T),cudaMemcpyDeviceToHost);
    delete[] buffer;
  }

  // most of the calls in the CPU version exist in the magma library in batched and unbatched forms. 
};

} // namespace qmcplusplus
#endif
