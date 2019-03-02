#ifndef EINSPLINE_TESTKERNEL_H
#define EINSPLINE_TESTKERNEL_H
#include "CUDA/GPUParams.h"
#include "Numerics/Einspline/multi_bspline_structs_cuda.h"

extern "C" void
launch_test_kernel(int num);

#endif
