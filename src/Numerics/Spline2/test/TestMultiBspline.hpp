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

#ifndef QMCPLUSPLUS_TEST_MULTI_BSPLINE_H
#define QMCPLUSPLUS_TEST_MULTI_BSPLINE_H

#include "Numerics/Spline2/BsplineAllocatorCUDA.hpp"

namespace qmcplusplus
{
using namespace einspline;

template<typename T,typename DT>
struct TestMultiBspline
{
  using CpuST = bspline_traits<Devices::CPU, T, 3>;
  using CudaST = bspline_traits<Devices::CUDA, DT, 3>;

  Allocator<Devices::CPU> cpu_allocator;
  Allocator<Devices::CUDA> cuda_allocator;
  typename CpuST::SplineType* cpu_spline;
  typename CudaST::SplineType* cuda_spline;
  int num_splines_;

  TestMultiBspline(int num_splines = 100) : cpu_spline(nullptr), cuda_spline(nullptr), num_splines_(num_splines) {};

  void create()
  {
    typename CpuST::BCType xBC;
    typename CpuST::BCType yBC;
    typename CpuST::BCType zBC;
    // You can't take references to the old school structs at least with gcc
    std::vector<typename CpuST::BCType*> bc_array = {&xBC,&yBC,&zBC};
    bc_code bCode = bc_code::PERIODIC;
    std::for_each(bc_array.begin(), bc_array.end(), [bCode](typename CpuST::BCType* bc)
        {
	  bc->lCode = bCode;
	  bc->rCode = bCode;
	  bc->lVal = 0.0;
	  bc->rVal = 10.0;
	});
    Ugrid xGrid;
    Ugrid yGrid;
    Ugrid zGrid;
    // You can't take references to the old school structs at least with gcc
    std::vector<Ugrid*> gr_array = {&xGrid, &yGrid, &zGrid};
    std::for_each(gr_array.begin(), gr_array.end(), [](Ugrid* ug)
        {
	  ug->start = 0.0;
	  ug->end = 10.0;
	  ug->num = 100;
	});

    cpu_allocator.allocateMultiBspline(cpu_spline, xGrid, yGrid, xGrid, xBC, yBC, zBC, num_splines_);
    REQUIRE( cpu_spline != nullptr );
    T dummyT;
    DT dummyDT;
    cuda_allocator.createMultiBspline_3d(cpu_spline, cuda_spline, dummyT, dummyDT);
    REQUIRE( cpu_spline != nullptr );
    REQUIRE( cuda_spline != nullptr );
  }

  void destroy()
  {
    cuda_allocator.destroy(cuda_spline);
    cpu_allocator.destroy(cpu_spline);
  }

};
}

#endif
