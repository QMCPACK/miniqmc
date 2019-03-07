#include "Numerics/Spline2/test/TestKernel.h"

__device__ double Bcuda[48];
__constant__ float  Acuda[48];

__global__ static void
test_kernel
(int N)
{
  int block = blockIdx.x;
  int thr   = threadIdx.x;
  int ir    = blockIdx.y;
  int off   = block*SPLINE_BLOCK_SIZE+thr;
  __shared__ double *myval;
  __shared__ double abc[64];
  __shared__ double3 r;
  __syncthreads();
}

__global__ static void
test_kernel_with_spline
(const double *coefs, int N)
{
  int block = blockIdx.x;
  int thr   = threadIdx.x;
  int ir    = blockIdx.y;
  int off   = block*SPLINE_BLOCK_SIZE+thr;
  __shared__ double *myval;
  __shared__ double abc[64];
  __shared__ double3 r;
  __syncthreads();
}

extern "C" void
launch_test_kernel(int N)
{
  dim3 dimBlock(SPLINE_BLOCK_SIZE);
  dim3 dimGrid(7, N);
  test_kernel<<<dimGrid,dimBlock>>>(N);
}

extern "C" void
launch_test_kernel_with_spline(multi_UBspline_3d_d<Devices::CUDA> *spline, int N)
{
  dim3 dimBlock(SPLINE_BLOCK_SIZE);
  dim3 dimGrid(7, N);
  test_kernel_with_spline<<<dimGrid,dimBlock>>>(spline->coefs, N);
  
}
