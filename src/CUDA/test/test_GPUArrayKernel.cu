////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2019 QMCPACK developers.
//
// File developed by:
// Peter Doak, doakpw@ornl.gov, Oak Ridge National Lab
//
// File created by:
// Peter Doak, doakpw@ornl.gov, Oak Ridge National Lab
////////////////////////////////////////////////////////////////////////////////

#include <cuda.h>
#include <cuda_runtime_api.h>

template<T>
__global__ static void
fill_GPUArray(<T>* start,  size_t pitch, int value, size_t width, size_t height)
{
  cudaMemset2D((void*)start, pitch, value, width, height);
}
