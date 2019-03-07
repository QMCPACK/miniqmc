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
  MultiBsplineFuncs<Devices::CUDA,double> mbf_CUDA;
  GPUArray<double,1,1> d_vals;
  d_vals.resize(1,128);
  d_vals.zero();
  std::vector<std::array<double,3>> pos = {{1,1,1}};
  mbf_CUDA.evaluate_v(tmb.cuda_spline, pos, d_vals.get_devptr(), 1, 128);
  MultiBsplineFuncs<Devices::CPU,double> mbf_CPU;
  std::vector<double> vals(128);
  mbf_CPU.evaluate_v(tmb.cpu_spline, pos, vals.data(), 128);
  aligned_vector<double> gpu_vals(128);
  d_vals.pull(gpu_vals);
  
  gpu.finalizeCUDAStreams();
}  



  // TEST_CAST("MultiBspline<CUDA> evaluate_v_value","[CUDA]")
  // {
  // Ugrid grid;
  // grid.start = 0.0;
  // grid.end = 1.0;
  // int N = 5;
  // grid.num = N;
  // double delta = (grid.end - grid.start)/grid.num;

  // double tpi = 2*M_PI;
  // double data[N*N*N];
  // for (int i = 0; i < N; i++) {
  //   for (int j = 0; j < N; j++) {
  //     for (int k = 0; k < N; k++) {
  //       double x = delta*i;
  //       double y = delta*j;
  //       double z = delta*k;
  //       data[N*N*i + N*j + k] = sin(tpi*x) + sin(3*tpi*y) + sin(4*tpi*z);
  //     }
  //   }
  // }

  // BCtype_d bc;
  // bc.lCode = PERIODIC;
  // bc.rCode = PERIODIC;
  
  
  // // UBspline_3d_d* s = create_UBspline_3d_d(grid, grid, grid, bc, bc, bc, data);
  // // REQUIRE(s);

  // // double val;
  // // eval_UBspline_3d_d(s, 0.0, 0.0, 0.0, &val);
  // // REQUIRE(val == Approx(0.0));

  // // double pos=0.99999999;
  // // eval_UBspline_3d_d(s, pos, pos, pos, &val);
  // // REQUIRE(val == Approx(0.0));

  // // eval_UBspline_3d_d(s, delta, delta, delta, &val);
  // // REQUIRE(val == Approx(data[N*N + N + 1]));

  // cpu_allocator.allocateMultiBspline(cpu_spline, grid, grid, grid, bc, bc, bc, 1)
  //   cpu_spline.set(
  
  // multi_UBspline_3d_d* m = create_multi_UBspline_3d_d(grid, grid, grid, bc, bc, bc, 1);
  // REQUIRE(m);

  // set_multi_UBspline_3d_d(m, 0, data);

  // eval_multi_UBspline_3d_d(m, delta, delta, delta, &val);
  // REQUIRE(val == Approx(data[N*N + N + 1]));


  // BCtype_s bc_s;
  // bc_s.lCode = PERIODIC;
  // bc_s.rCode = PERIODIC;

  // multi_UBspline_3d_s* ms = create_multi_UBspline_3d_s(grid, grid, grid, bc_s, bc_s, bc_s, 1);
  // REQUIRE(ms);
  // set_multi_UBspline_3d_s_d(ms, 0, data);

  // float fval;
  // eval_multi_UBspline_3d_s(ms, delta, delta, delta, &fval);
  // REQUIRE(fval == Approx(data[N*N + N + 1]));

  // float grads[3];
  // float hess[9];
  // eval_multi_UBspline_3d_s_vgh(ms, 0.1, 0.2, 0.0, &fval, grads, hess);

  // // See miniqmc-python/splines/test_3d.py for values
  // REQUIRE(grads[0] == Approx(5.11104213833));
  // REQUIRE(grads[1] == Approx(5.98910634201));
  // REQUIRE(grads[2] == Approx(-6.17832080889));

  // // All off-diagonal values of the Hessian for this data should be zero
  // REQUIRE(hess[1] == Approx(0.0));
  // REQUIRE(hess[2] == Approx(0.0));

  // REQUIRE(hess[3] == Approx(0.0));
  // REQUIRE(hess[5] == Approx(0.0));

  // REQUIRE(hess[6] == Approx(0.0));
  // REQUIRE(hess[7] == Approx(0.0));
  // }
  
}
