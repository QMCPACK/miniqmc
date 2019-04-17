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
#include <boost/tuple/tuple.hpp>
#include <boost/iterator/zip_iterator.hpp>

#include "CUDA/GPUArray.h"
#include "CUDA/GPUParams.h"
#include "Numerics/Containers.h"
#include "Numerics/Spline2/MultiBsplineFuncsCUDA.hpp"
#include "Numerics/Spline2/MultiBsplineFuncs.hpp"
#include "Numerics/Spline2/test/TestMultiBspline.hpp"
#include "Numerics/Spline2/test/CheckMultiBsplineEvalOutput.hpp"

namespace qmcplusplus
{
TEST_CASE("MultiBspline<CUDA> instantiation", "[CUDA][Spline2]")
{
  //can I infer type like this?
  MultiBsplineFuncs<Devices::CUDA, double> mbO;
  //Devices::CUDA, double> mbcd;
  //MultiBspline<Devices::CUDA, float> mbcf;
}

TEST_CASE("MultiBspline<CUDA> single block evaluate_v", "[CUDA][Spline2]")
{
  Gpu& gpu = Gpu::get();
  TestMultiBspline<double, double> tmb(64);
  tmb.create();
  MultiBsplineFuncs<Devices::CUDA, double> mbf_CUDA;
  GPUArray<double, 1, 1> d_vals;
  d_vals.resize(1, 64);
  d_vals.zero();
  std::vector<std::array<double, 3>> pos = {{1, 1, 1}};
  mbf_CUDA.evaluate_v(tmb.cuda_spline, pos, d_vals.get_devptr(), 1, 64, 64);
  MultiBsplineFuncs<Devices::CPU, double> mbf_CPU;
  aligned_vector<double> vals(64);
  mbf_CPU.evaluate_v(tmb.cpu_splines[0], pos, vals.data(), 64);
  aligned_vector<double> gpu_vals(64);
  d_vals.pull(gpu_vals);

  bool matching_spline_vals = true;
  for (int i = 0; i < gpu_vals.size(); ++i)
  {
    //std::cout << vals[i] << " : " << gpu_vals[i] << '\n';
    if (vals[i] != Approx(gpu_vals[i]).epsilon(0.005))
    {
      bool matching_spline_vals = false;
      std::cout << "evaluation values do not match (cpu : gpu)  " << vals[i] << " : " << gpu_vals[i]
                << '\n';
      break;
    }
  }
  REQUIRE(matching_spline_vals);
}


TEST_CASE("MultiBspline<CUDA> single block evaluate_vgh", "[CUDA][Spline2]")
{
  using T  = double;
  Gpu& gpu = Gpu::get();
  TestMultiBspline<double, double> tmb(64);
  tmb.create();
  MultiBsplineFuncs<Devices::CUDA, double> mbf_CUDA;
  GPUArray<double, 1, 1> d_vals;
  GPUArray<double, 3, 1> d_grads;
  GPUArray<double, 6, 1> d_hess;
  d_vals.resize(1, 64);
  d_vals.zero();
  d_grads.resize(1, 64);
  d_grads.zero();
  d_hess.resize(1, 64);
  d_hess.zero();
  std::vector<std::array<double, 3>> pos = {{1, 1, 1}};
  cudaStream_t stream                    = cudaStreamPerThread;
  mbf_CUDA.evaluate_vgh(tmb.cuda_spline, pos, d_vals.get_devptr(), d_grads.get_devptr(),
                        d_hess.get_devptr(), 1, 64, 1, stream);
  aligned_vector<T> gpu_vals(64);
  VectorSoAContainer<T, 3> gpu_grads(64);
  VectorSoAContainer<T, 6> gpu_hess(64);
  d_vals.pull(gpu_vals);
  d_grads.pull(gpu_grads);
  d_hess.pull(gpu_hess);
  MultiBsplineFuncs<Devices::CPU, T> mbf_CPU;
  aligned_vector<T> vals(64);
  VectorSoAContainer<T, 3> grads(64);
  VectorSoAContainer<T, 6> hess(64);
  mbf_CPU.evaluate_vgh(tmb.cpu_splines[0], pos, vals.data(), grads.data(), hess.data(), 64);

  CheckMultiBsplineEvalOutput<T> check_eval;
  bool eval_checks = check_eval(vals, gpu_vals, grads, gpu_grads, hess, gpu_hess);
  REQUIRE(eval_checks);
}

TEST_CASE("MultiBspline<CUDA> multi block evaluate_vgh", "[CUDA][Spline2]")
{
  using T  = double;
  Gpu& gpu = Gpu::get();
  TestMultiBspline<T, T> tmb(128, 2, 10);
  tmb.create();
  MultiBsplineFuncs<Devices::CUDA, double> mbf_CUDA;
  GPUArray<double, 1, 1> d_vals;
  GPUArray<double, 3, 1> d_grads;
  GPUArray<double, 6, 1> d_hess;
  d_vals.resize(1, 128);
  d_vals.zero();
  d_grads.resize(1, 128);
  d_grads.zero();
  d_hess.resize(1, 128);
  d_hess.zero();
  std::vector<std::array<double, 3>> pos = {{1, 1, 1}};
  cudaStream_t stream                    = cudaStreamPerThread;
  mbf_CUDA.evaluate_vgh(tmb.cuda_spline, pos, d_vals.get_devptr(), d_grads.get_devptr(),
                        d_hess.get_devptr(), 2, 64, 64, stream);
  aligned_vector<T> gpu_vals(128);
  VectorSoAContainer<T, 3> gpu_grads(128);
  VectorSoAContainer<T, 6> gpu_hess(128);
  d_vals.pull(gpu_vals);
  d_grads.pull(gpu_grads);
  d_hess.pull(gpu_hess);
  MultiBsplineFuncs<Devices::CPU, T> mbf_CPU;
  aligned_vector<T> vals(128);
  VectorSoAContainer<T, 3> grads(128);
  VectorSoAContainer<T, 6> hess(128);
  mbf_CPU.evaluate_vgh(tmb.cpu_splines[0], pos, vals.data(), grads.data(), hess.data(), 64);
  mbf_CPU.evaluate_vgh(tmb.cpu_splines[1], pos, &(vals.data()[64]), &(grads.data()[64]),
                       &(hess.data()[64]), 64);


  CheckMultiBsplineEvalOutput<T> check_eval;
  bool eval_checks = check_eval(vals, gpu_vals, grads, gpu_grads, hess, gpu_hess);
  REQUIRE(eval_checks);
}


/** This belongs in test of BsplineAllocatorCUDA but its convenient here.
 */
TEST_CASE("MultiBspline<CUDA> multi block coefficients", "[CUDA][Spline2]")
{
  TestMultiBspline<double, double> tmb(64, 2, 10);
  tmb.create();
  size_t size_of_all_coefs = (tmb.cpu_splines[0]->x_grid.num + 3) *
      (tmb.cpu_splines[0]->y_grid.num + 3) * (tmb.cpu_splines[0]->z_grid.num + 3);
  int num_splines =
      std::accumulate(tmb.cpu_splines.begin(), tmb.cpu_splines.end(), 0,
                      [](int a, typename bspline_traits<Devices::CPU, double, 3>::SplineType* spl) {
                        return a + spl->num_splines;
                      });
  std::vector<double> coefs;
  size_t coefs_size = size_of_all_coefs * num_splines;
  coefs.resize(coefs_size);

  cudaError_t cu_err = cudaMemcpy(coefs.data(), tmb.cuda_spline->coefs, coefs_size * sizeof(double),
                                  cudaMemcpyDeviceToHost);
  std::vector<double> cpu_coefs;
  cpu_coefs.resize(coefs_size);
  REQUIRE(cu_err == cudaError::cudaSuccess);
  REQUIRE(coefs[tmb.cpu_splines[0]->num_splines] == tmb.cpu_splines[1]->coefs[0]);
}

TEST_CASE("MultiBspline<CUDA> multi block evaluate_v", "[CUDA][Spline2]")
{
  Gpu& gpu = Gpu::get();
  TestMultiBspline<double, double> tmb(64, 4);
  tmb.create();
  MultiBsplineFuncs<Devices::CUDA, double> mbf_CUDA;
  GPUArray<double, 1, 1> d_vals;
  d_vals.resize(1, 128);
  d_vals.zero();
  std::vector<std::array<double, 3>> pos = {{1, 1, 1}};
  mbf_CUDA.evaluate_v(tmb.cuda_spline, pos, d_vals.get_devptr(), 1, 128, 128);
  aligned_vector<double> gpu_vals(128);
  d_vals.pull(gpu_vals);

  MultiBsplineFuncs<Devices::CPU, double> mbf_CPU;
  aligned_vector<double> vals(128);
  mbf_CPU.evaluate_v(tmb.cpu_splines[0], pos, vals.data(), 64);
  mbf_CPU.evaluate_v(tmb.cpu_splines[1], pos, &(vals.data()[64]), 64);

  bool matching_spline_vals = true;
  for (int i = 0; i < gpu_vals.size(); ++i)
  {
    //std::cout << vals[i] << " : " << gpu_vals[i] << '\n';
    if (vals[i] != Approx(gpu_vals[i]).epsilon(0.005))
    {
      bool matching_spline_vals = false;
      std::cout << "evaluation values do not match (cpu : gpu)  " << vals[i] << " : " << gpu_vals[i]
                << '\n';
      break;
    }
  }

  REQUIRE(matching_spline_vals);
}

/** needs to actually check values but it is also nice to see it not crash
 */
// TEST_CASE("MultiBspline<CUDA> multi pos evaluate_v", "[CUDA][Spline2]")
// {
//   Gpu& gpu = Gpu::get();
//   TestMultiBspline<double, double> tmb(64, 4);
//   tmb.create();
//   MultiBsplineFuncs<Devices::CUDA, double> mbf_CUDA;
//   GPUArray<double, 1, 1> d_vals;
//   std::vector<std::array<double, 3>> pos = {{1, 1, 1},{0.24,0.24,0.24},{1, 1, 1},{0.24,0.24,0.24}};
//   d_vals.resize(1, 256 * pos.size());
//   d_vals.zero();

//   mbf_CUDA.evaluate_v(tmb.cuda_spline, pos, d_vals.get_devptr(), 1, 256);
//   aligned_vector<double> gpu_vals(256 * pos.size());
//   d_vals.pull(gpu_vals);

//   bool matching_spline_vals = true;
// }


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

} // namespace qmcplusplus
