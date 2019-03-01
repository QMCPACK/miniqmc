#include "Numerics/Einspline/test/TestKernel.h"

__device__ double Bcuda[48];
__constant__ float  Acuda[48];

extern "C"
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

extern "C" void
launch_test_kernel(int N)
{
  dim3 dimBlock(SPLINE_BLOCK_SIZE);
  dim3 dimGrid(7, N);
  test_kernel<<<dimGrid,dimBlock>>>(N);
}