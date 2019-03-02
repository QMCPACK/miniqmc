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

#include "catch.hpp"
#include "CUDA/GPUArray.h"
#include "CUDA/GPUParams.h"
#include "Numerics/Spline2/MultiBsplineFuncsCUDA.hpp"
#include "Numerics/Spline2/MultiBsplineFuncs.hpp"
#include "Numerics/Spline2/test/TestMultiBspline.hpp"
namespace qmcplusplus
{

TEST_CASE("MultiBspline<CUDA> instantiation", "[CUDA][Spline2]")
{
  //can I infer type like this?
  MultiBsplineFuncs<Devices::CUDA,double> mbO;
  //Devices::CUDA, double> mbcd;
  //MultiBspline<Devices::CUDA, float> mbcf;  
}

TEST_CASE("MultiBspline<CUDA> evaluate_v", "[CUDA][Spline2]")
{
  Gpu& gpu = Gpu::get();
  gpu.initCUDAStreams();
  TestMultiBspline<double, double> tmb(128);
  tmb.create();
  MultiBsplineFuncs<Devices::CUDA,double> mbO;
  GPUArray<double,1,1> d_vals;
  d_vals.resize(sizeof(double)* 16,sizeof(double) * 16);
  d_vals.zero();
  std::vector<std::array<double,3>> pos = {{0,0,0},{0,1,0}};
  double* vals = d_vals[0];
  mbO.evaluate_v(tmb.cuda_spline, pos, vals, 1);
  gpu.finalizeCUDAStreams();
}  

}
