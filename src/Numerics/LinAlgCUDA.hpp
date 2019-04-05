#ifndef QMCPLUSPLUS_LINALG_CUDA_H
#define QMCPLUSPLUS_LINALG_CUDA_H

#include "cuda.h"
#include "cublas_v2.h"
#include "LinAlgCPU.h"

#include "CUDA/PinnedHostBuffer.hpp"

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
		  const T* x, int incx, T beta, T* y, int incy, cudaStream_t stream);

template<>
inline void TGemV(int n, int m, double alpha, const double *restrict amat, int lda,
		  const double* x, int incx, double beta, double* y, int incy, cudaStream_t stream)
{
    cublasSetStream(HandleToCublas::get(),stream);
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
		const T *y, int incy, T *a, int lda, cudaStream_t stream);

template<>
inline void Ger(int m, int n, double alpha, const double *x, int incx,
		const double *y, int incy, double *a, int lda, cudaStream_t stream)
{
  cublasSetStream(HandleToCublas::get(),stream);
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
  updateRow(Matrix<T>& psiMinv, aligned_vector<T>& psiV, int m, int lda, int rowchanged, RT c_ratio_in, size_t cuda_buffer_size, void* cuda_buffer, PinnedHostBuffer& host_buffer, cudaStream_t stream)
  {
    constexpr T cone(1);
    constexpr T czero(0);
    T temp[m];
    T rcopy[m];
    T* dev_ptr = static_cast<T*>(cuda_buffer);
    //char* buffer = new char[cuda_buffer_size];
    T* psi_v_dev_ptr = (double*)dev_ptr + psiMinv.size(); 
    T* temp_dev_ptr = psi_v_dev_ptr + psiV.size();

    host_buffer(stream);
    //host_buffer.fromNormalTcpy(psiMinv.data(), 0, psiMinv.size());
    host_buffer.fromNormalTcpy(psiV.data(), psiMinv.size(), psiV.size());
    host_buffer.partialToDevice(psi_v_dev_ptr, psiMinv.size(), psiV.size());    
    T c_ratio = cone / c_ratio_in;
    linalg::TGemV(m, m, c_ratio, dev_ptr, m, psi_v_dev_ptr, 1, czero, temp_dev_ptr, 1, stream);
    host_buffer.partialFromDevice(temp_dev_ptr, psiMinv.size() + psiV.size(), m);
    //cudaMemcpyAsync((void*)buffer, dev_ptr, cuda_buffer_size, cudaMemcpyDefault, stream);
    cudaStreamSynchronize(stream);
    host_buffer.toNormalTcpy(temp, psiMinv.size() + psiV.size(), m );
    cudaStreamSynchronize(stream);
    temp[rowchanged] = cone - c_ratio;
    std::copy_n(psiMinv.data() + m * rowchanged, m, rcopy);
    host_buffer.fromNormalTcpy(temp, psiMinv.size() + psiV.size(), m );
    host_buffer.fromNormalTcpy(rcopy, psiMinv.size() + psiV.size() + m, m);
    host_buffer.partialToDevice(temp_dev_ptr, psiMinv.size() + psiV.size(), 2 * m);
    linalg::Ger(m, m, -cone, temp_dev_ptr+m, 1, temp_dev_ptr, 1, dev_ptr, m, stream);
    host_buffer.partialFromDevice(dev_ptr, 0, psiMinv.size());
    cudaStreamSynchronize(stream);
    host_buffer.toNormalTcpy(psiMinv.data(), 0, psiMinv.size());
  }

  // most of the calls in the CPU version exist in the magma library in batched and unbatched forms. 
};

} // namespace qmcplusplus
#endif
