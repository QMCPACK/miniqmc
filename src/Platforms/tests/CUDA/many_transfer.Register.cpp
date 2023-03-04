#include "catch.hpp"

#include <CUDA/CUDAruntime.hpp>
#include <vector>
#include <cstdlib>
#include <iostream>
#include <string>
#include <sstream>
#include <stdexcept>
#include <timer.h>

#define N 32768
#define SEG_SIZE 1024

TEST_CASE("many_transfer.Register", "[CUDA]")
{
  int count = 0;
  cudaCheck(cudaGetDeviceCount(&count));
  if (count == 0)
    std::exit(100);

  std::vector<double*> segments(N, nullptr);
  std::vector<double*> segments_dev(N, nullptr);
  for (int i = 0; i < N; i++)
  {
    //cudaCheck(cudaMallocHost(&segments[i], SEG_SIZE * sizeof(double)));
    segments[i] = (double*)malloc(SEG_SIZE * sizeof(double));
    cudaCheck(cudaHostRegister(segments[i], SEG_SIZE * sizeof(double), cudaHostRegisterDefault));
    cudaCheck(cudaMalloc(&segments_dev[i], SEG_SIZE * sizeof(double)));
  }

  cudaStream_t stream;
  cudaCheck(cudaStreamCreate(&stream));

  { //warmup
    for (int i = 0; i < N; i++)
      cudaCheck(
          cudaMemcpyAsync(segments_dev[i], segments[i], SEG_SIZE * sizeof(double), cudaMemcpyHostToDevice, stream));
    cudaCheck(cudaStreamSynchronize(stream));
  }

  cudaCheck(cudaStreamSynchronize(stream));
  {
    Timer local("many_transfer.Register");
    for (int i = 0; i < N; i++)
      cudaCheck(
          cudaMemcpyAsync(segments_dev[i], segments[i], SEG_SIZE * sizeof(double), cudaMemcpyHostToDevice, stream));
  }
  cudaCheck(cudaStreamSynchronize(stream));
  std::cout << "Success" << std::endl;

  cudaCheck(cudaStreamDestroy(stream));

  for (int i = 0; i < N; i++)
  {
    //cudaCheck(cudaFreeHost(segments[i]));
    cudaCheck(cudaHostUnregister(segments[i]));
    free(segments[i]);
    cudaCheck(cudaFree(segments_dev[i]));
  }
}
