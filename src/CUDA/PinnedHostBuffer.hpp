#ifndef QMCPLUSPLUS_CUDA_PINNED_HOST_BUFFER
#define QMCPLUSPLUS_CUDA_PINNED_HOST_BUFFER

#include <array>
#include <vector>
#include <string>
#include <stdexcept>
#include <cuda_runtime_api.h>

struct PinnedHostBuffer
  {
     PinnedHostBuffer() : stream_(cudaStreamPerThread), buffer_(nullptr), byte_size_(0) {}
     PinnedHostBuffer(cudaStream_t& stream) : stream_(stream), buffer_(nullptr), byte_size_(0) {}
     
    ~PinnedHostBuffer()
    {
      if (buffer_ != nullptr) cudaFreeHost(buffer_);
    }

    void operator() (cudaStream_t& stream) { stream_ = stream; }

    void resize(size_t size)
    {
      if (size != byte_size_)
      {
	  if (buffer_ != nullptr) cudaFreeHost(buffer_);
	  cudaMallocHost((void**)&buffer_, size );
	  byte_size_ = size;
      }
    }

  void cudaCheck(cudaError_t cu_err)
	  {
	      if (cu_err != cudaError::cudaSuccess)
	      {
		  std::string error_string(cudaGetErrorName(cu_err));
		  error_string += ": ";
		  error_string += cudaGetErrorString(cu_err);
		  throw std::runtime_error(error_string);
	      }
	  }
	      
  void copyFromDevice(void* dev_ptr)
  {
    cudaCheck(cudaMemcpyAsync(buffer_, dev_ptr, byte_size_, cudaMemcpyDefault, stream_));    
  }

  void copyToNormalMem(void* buffer)
  {
      cudaCheck(cudaMemcpyAsync(buffer, buffer_, byte_size_, cudaMemcpyDefault, stream_));
  }

  
  template<typename T>
  void toNormalMemcpy(T* buffer, int offset_T, size_t count)
  {
      cudaMemcpyAsync(buffer, buffer_ + offset_T * sizeof(T), count * sizeof(T), cudaMemcpyDefault, stream_);
  }

  private:
    void* buffer_;
    size_t byte_size_;
    cudaStream_t stream_;
  };
#endif
