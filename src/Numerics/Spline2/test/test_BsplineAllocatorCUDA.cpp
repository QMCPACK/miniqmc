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
#include "Numerics/Spline2/BsplineAllocatorCUDA.hpp"
#include "Numerics/Spline2/MultiBsplineFuncs.hpp"
#include "Numerics/Spline2/bspline_traits.hpp"
#include "Numerics/Einspline/bspline_structs.h"
#include "Numerics/Einspline/bspline_base.h"
#include "Numerics/Einspline/multi_bspline_structs_cuda.h"
#include "Numerics/Spline2/test/TestMultiBspline.hpp"

namespace qmcplusplus
{
using namespace einspline;


TEST_CASE("Allocator<CUDA> instantiation", "[CUDA]")
{
  Allocator<Devices::CUDA> cuda_allocator;
}

TEST_CASE("Allocator<CUDA> Multibspline double", "[CUDA][Spline2]")
{
  TestMultiBspline<double, double> tmb;
  tmb.create();
  tmb.destroy();
}

TEST_CASE("Allocator<CUDA> Multibspline float", "[CUDA][Spline2]")
{
  TestMultiBspline<float, float> tmb;
  tmb.create();
  tmb.destroy();
}

TEST_CASE("Allocator<CUDA> Multibspline mixed", "[CUDA][Spline2]")
{
  TestMultiBspline<double, float> tmb;
  tmb.create();
  tmb.destroy();
}


}
