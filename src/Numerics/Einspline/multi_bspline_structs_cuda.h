#ifndef MULTI_BSPLINE_STRUCTS_CUDA_H
#define MULTI_BSPLINE_STRUCTS_CUDA_H

#include <cuda.h>
#include "multi_bspline_structs.h"

/** @file
 *  The cuda splines are not self contained
 *  there is an implied host side multi_UBspline_3d_x<Devices:CPU>
 */

template<>
struct multi_UBspline_3d_s<Devices::CUDA>
{
  float* coefs;
  uint3 stride;
  float3 gridInv;
  unit3 dim;
  int num_splines;
  int num_split_splines;
};

template<>
struct multi_UBspline_3d_d<Devices::CUDA>
{
  double* coefs;
  uint3 stride;
  double3 gridInv;
  unit3 dim;
  int num_splines;
  int num_split_splines;
};

#endif
