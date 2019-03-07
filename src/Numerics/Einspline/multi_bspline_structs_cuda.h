#ifndef MULTI_BSPLINE_STRUCTS_CUDA_H
#define MULTI_BSPLINE_STRUCTS_CUDA_H

#include <cuda.h>
#include <vector_types.h>
#include "multi_bspline_structs.h"

#define SPLINE_BLOCK_SIZE 16

/** @file
 *  The cuda splines are not self contained i.e. information is missing
 */

template<>
struct multi_UBspline_3d_s<Devices::CUDA>
{
  float* coefs;
  uint3 stride;
  float3 gridInv;
  uint3 dim;
  int num_splines;
  int num_split_splines;
};

template<>
struct multi_UBspline_3d_d<Devices::CUDA>
{
  double* Bcuda;
  double* coefs;
  uint3 stride;
  double3 gridInv;
  uint3 dim;
  int num_splines;
  int num_split_splines;
};

#endif
