//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by: Jeremy McMinnis, jmcminis@gmail.com, University of Illinois at Urbana-Champaign
//		      Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//
// File created by: Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//////////////////////////////////////////////////////////////////////////////////////


#include <stdio.h>
#include <stdexcept>
#include "CUDA/GPUParams.h"
#include "CUDA/CUDA_const.h"
#include "multi_bspline_structs_cuda.h"

multi_UBspline_3d_s<Devices::CUDA>*
create_multi_UBspline_3d_s_cuda(multi_UBspline_3d_s<Devices::CPU>* spline, int blocked_size)
{
  // clang-format off
  float A_h[48] = {-1.0 / 6.0, 3.0 / 6.0,  -3.0 / 6.0, 1.0 / 6.0,
		   3.0 / 6.0, -6.0 / 6.0, 0.0 / 6.0, 4.0 / 6.0,
		   -3.0 / 6.0, 3.0 / 6.0,  3.0 / 6.0, 1.0 / 6.0,
		   1.0 / 6.0,  0.0 / 6.0, 0.0 / 6.0,  0.0 / 6.0,
		   0.0,        -0.5,      1.0,       -0.5,
		   0.0,        1.5,        -2.0,       0.0,
		   0.0,       -1.5,      1.0,        0.5,
                   0.0,        0.5,        0.0,        0.0,
		   0.0,       0.0,        -1.0,        1.0,
		   0.0,        0.0,        3.0,       -2.0,
		   0.0,        0.0,       -3.0,       1.0,
		   0.0,        0.0,       1.0,       0.0};
  // clang-format on
  
  Gpu& gpu      = Gpu::get();
  if (gpu.device_group_size > 1)
  {
    for (unsigned int i = 0; i < gpu.device_group_size; i++)
    {
      cudaSetDevice(gpu.device_group_numbers[i]);
      cudaMemcpyToSymbol(Acuda, A_h, 48 * sizeof(float), 0, cudaMemcpyHostToDevice);
    }
    cudaSetDevice(gpu.device_group_numbers[gpu.relative_rank % gpu.device_group_size]);
  }
  else
    cudaMemcpyToSymbol(Acuda, A_h, 48 * sizeof(float), 0, cudaMemcpyHostToDevice);

  multi_UBspline_3d_s<Devices::CUDA>* cuda_spline =
      (multi_UBspline_3d_s<Devices::CUDA>*)malloc(sizeof(multi_UBspline_3d_s<Devices::CUDA>));

  cuda_spline->num_splines       = spline->num_splines;
  cuda_spline->num_split_splines = spline->num_splines;

  int Nx = spline->x_grid.num + 3;
  int Ny = spline->y_grid.num + 3;
  int Nz = spline->z_grid.num + 3;

  int N            = spline->num_splines;
  int spline_start = 0;
  if (gpu.device_group_size > 1)
  {
    if (N < gpu.device_group_size)
    {
      if (gpu.rank == 0)
        fprintf(stdout, "Error: Not enough splines (%i) to split across %i devices.\n", N,
                gpu.device_group_size);
      abort();
    }
    N += gpu.device_group_size - 1;
    N /= gpu.device_group_size;
    spline_start = N * (gpu.relative_rank % gpu.device_group_size);
#ifdef SPLIT_SPLINE_DEBUG
    fprintf(stderr, "splines %i - %i of %i\n", spline_start, spline_start + N, spline->num_splines);
#endif
  }
#ifdef SPLIT_SPLINE_DEBUG
  else
    fprintf(stderr, "splines N = %i\n", N);
#endif
  cuda_spline->num_split_splines = N;
  int num_splines                = N;
  if ((N % COALLESCED_SIZE) != 0) N += COALLESCED_SIZE - (N % COALLESCED_SIZE);

  cuda_spline->stride.x = Ny * Nz * N;
  cuda_spline->stride.y = Nz * N;
  cuda_spline->stride.z = N;

  cuda_spline->gridInv.x = spline->x_grid.delta_inv;
  cuda_spline->gridInv.y = spline->y_grid.delta_inv;
  cuda_spline->gridInv.z = spline->z_grid.delta_inv;

  cuda_spline->dim.x = spline->x_grid.num;
  cuda_spline->dim.y = spline->y_grid.num;
  cuda_spline->dim.z = spline->z_grid.num;

  size_t size = Nx * Ny * Nz * N * sizeof(float);
  size        = (((size + 15) / 16) * 16);
  cudaMalloc((void**)&(cuda_spline->coefs), size);

  float* spline_buff = (float*)malloc(size);
  if (!spline_buff)
  {
    fprintf(stderr, "Failed to allocate memory for temporary spline buffer.\n");
    abort();
  }

  for (int ix = 0; ix < Nx; ix++)
    for (int iy = 0; iy < Ny; iy++)
      for (int iz = 0; iz < Nz; iz++)
        for (int isp = 0; isp < N; isp++)
        {
          int curr_sp = isp + spline_start;
          if ((curr_sp < spline->num_splines) && (isp < num_splines))
          {
            spline_buff[ix * cuda_spline->stride.x + iy * cuda_spline->stride.y +
                        iz * cuda_spline->stride.z + isp] =
                spline->coefs[ix * spline->x_stride + iy * spline->y_stride +
                              iz * spline->z_stride + curr_sp];
          }
          else
            spline_buff[ix * cuda_spline->stride.x + iy * cuda_spline->stride.y +
                        iz * cuda_spline->stride.z + isp] = 0.0;
        }
  cudaMemcpy(cuda_spline->coefs, spline_buff, size, cudaMemcpyHostToDevice);

  free(spline_buff);

  return cuda_spline;
}


multi_UBspline_3d_s<Devices::CUDA>*
create_multi_UBspline_3d_s_cuda_conv(multi_UBspline_3d_d<Devices::CPU>* spline, int blocked_size)
{
  // fprintf (stderr, "In create_multi_UBspline_3d_s_cuda_conv.\n");
  float A_h[48] = {-1.0 / 6.0, 3.0 / 6.0,  -3.0 / 6.0, 1.0 / 6.0, 3.0 / 6.0, -6.0 / 6.0, 0.0 / 6.0,
                   4.0 / 6.0,  -3.0 / 6.0, 3.0 / 6.0,  3.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0,  0.0 / 6.0,
                   0.0 / 6.0,  0.0 / 6.0,  0.0,        -0.5,      1.0,       -0.5,       0.0,
                   1.5,        -2.0,       0.0,        0.0,       -1.5,      1.0,        0.5,
                   0.0,        0.5,        0.0,        0.0,       0.0,       0.0,        -1.0,
                   1.0,        0.0,        0.0,        3.0,       -2.0,      0.0,        0.0,
                   -3.0,       1.0,        0.0,        0.0,       1.0,       0.0};
  Gpu& gpu      = Gpu::get();
  if (gpu.device_group_size > 1)
  {
    for (unsigned int i = 0; i < gpu.device_group_size; i++)
    {
      cudaSetDevice(gpu.device_group_numbers[i]);
      cudaMemcpyToSymbol(Acuda, A_h, 48 * sizeof(float), 0, cudaMemcpyHostToDevice);
    }
    cudaSetDevice(gpu.device_group_numbers[gpu.relative_rank % gpu.device_group_size]);
  }
  else
    cudaMemcpyToSymbol(Acuda, A_h, 48 * sizeof(float), 0, cudaMemcpyHostToDevice);

  multi_UBspline_3d_s<Devices::CUDA>* cuda_spline =
      (multi_UBspline_3d_s<Devices::CUDA>*)malloc(sizeof(multi_UBspline_3d_s<Devices::CUDA>));

  cuda_spline->num_splines       = spline->num_splines;
  cuda_spline->num_split_splines = spline->num_splines;

  int Nx = spline->x_grid.num + 3;
  int Ny = spline->y_grid.num + 3;
  int Nz = spline->z_grid.num + 3;

  int N            = spline->num_splines;
  int spline_start = 0;
  if (gpu.device_group_size > 1)
  {
    if (N < gpu.device_group_size)
    {
      if (gpu.rank == 0)
        fprintf(stdout, "Error: Not enough splines (%i) to split across %i devices.\n", N,
                gpu.device_group_size);
      abort();
    }
    N += gpu.device_group_size - 1;
    N /= gpu.device_group_size;
    spline_start = N * (gpu.relative_rank % gpu.device_group_size);
#ifdef SPLIT_SPLINE_DEBUG
    fprintf(stderr, "splines %i - %i of %i\n", spline_start, spline_start + N, spline->num_splines);
#endif
  }
#ifdef SPLIT_SPLINE_DEBUG
  else
    fprintf(stderr, "splines N = %i\n", N);
#endif
  cuda_spline->num_split_splines = N;
  int num_splines                = N;
  if ((N % COALLESCED_SIZE) != 0) N += COALLESCED_SIZE - (N % COALLESCED_SIZE);

  cuda_spline->stride.x = Ny * Nz * N;
  cuda_spline->stride.y = Nz * N;
  cuda_spline->stride.z = N;

  cuda_spline->gridInv.x = spline->x_grid.delta_inv;
  cuda_spline->gridInv.y = spline->y_grid.delta_inv;
  cuda_spline->gridInv.z = spline->z_grid.delta_inv;

  cuda_spline->dim.x = spline->x_grid.num;
  cuda_spline->dim.y = spline->y_grid.num;
  cuda_spline->dim.z = spline->z_grid.num;

  size_t size = Nx * Ny * Nz * N * sizeof(float);

  cudaError_t err = cudaMalloc((void**)&(cuda_spline->coefs), size);
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to allocate GPU spline coefs memory.  Error:  %s\n",
            cudaGetErrorString(err));
    abort();
  }
  float* spline_buff = (float*)malloc(size);
  if (!spline_buff)
  {
    fprintf(stderr, "Failed to allocate memory for temporary spline buffer.\n");
    abort();
  }

  for (int ix = 0; ix < Nx; ix++)
    for (int iy = 0; iy < Ny; iy++)
      for (int iz = 0; iz < Nz; iz++)
        for (int isp = 0; isp < N; isp++)
        {
          int curr_sp = isp + spline_start;
          if ((curr_sp < spline->num_splines) && (isp < num_splines))
          {
            spline_buff[ix * cuda_spline->stride.x + iy * cuda_spline->stride.y +
                        iz * cuda_spline->stride.z + isp] =
                spline->coefs[ix * spline->x_stride + iy * spline->y_stride +
                              iz * spline->z_stride + curr_sp];
          }
          else
            spline_buff[ix * cuda_spline->stride.x + iy * cuda_spline->stride.y +
                        iz * cuda_spline->stride.z + isp] = 0.0;
        }
  err = cudaMemcpy((void*)cuda_spline->coefs, (void*)spline_buff, size, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  //err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to copy spline to GPU memory.  Error:  %s\n", cudaGetErrorString(err));
    abort();
  }
  free(spline_buff);

  return cuda_spline;
}


multi_UBspline_3d_d<Devices::CUDA>*
create_multi_UBspline_3d_d_cuda(multi_UBspline_3d_d<Devices::CPU>* spline, int blocked_size)
{
  double B_h[48] = {-1.0 / 6.0, 3.0 / 6.0,  -3.0 / 6.0, 1.0 / 6.0, 3.0 / 6.0, -6.0 / 6.0, 0.0 / 6.0,
                    4.0 / 6.0,  -3.0 / 6.0, 3.0 / 6.0,  3.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0,  0.0 / 6.0,
                    0.0 / 6.0,  0.0 / 6.0,  0.0,        -0.5,      1.0,       -0.5,       0.0,
                    1.5,        -2.0,       0.0,        0.0,       -1.5,      1.0,        0.5,
                    0.0,        0.5,        0.0,        0.0,       0.0,       0.0,        -1.0,
                    1.0,        0.0,        0.0,        3.0,       -2.0,      0.0,        0.0,
                    -3.0,       1.0,        0.0,        0.0,       1.0,       0.0};
  Gpu& gpu       = Gpu::get();
  multi_UBspline_3d_d<Devices::CUDA>* cuda_spline = new multi_UBspline_3d_d<Devices::CUDA>;
  cudaError_t err;
  if (gpu.device_group_size > 1)
  {
    for (unsigned int i = 0; i < gpu.device_group_size; i++)
    {
      cudaSetDevice(gpu.device_group_numbers[i]);
      cudaMalloc((void**)&cuda_spline->Bcuda, 48 * sizeof(double));
      err = cudaMemcpy(cuda_spline->Bcuda, B_h, 48 * sizeof(double), cudaMemcpyHostToDevice);
    }
    cudaSetDevice(gpu.device_group_numbers[gpu.relative_rank % gpu.device_group_size]);
  }
  else
  {
    cudaMalloc((void**)&cuda_spline->Bcuda, 48 * sizeof(double));
    err = cudaMemcpy(cuda_spline->Bcuda, B_h, 48 * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaError::cudaSuccess)
      throw std::runtime_error("Failed GPU allocation");
  }

  cuda_spline->num_splines       = spline->num_splines;
  cuda_spline->num_split_splines = spline->num_splines;

  int Nx = spline->x_grid.num + 3;
  int Ny = spline->y_grid.num + 3;
  int Nz = spline->z_grid.num + 3;

  cuda_spline->stride.x = Ny * Nz * blocked_size;
  cuda_spline->stride.y = Nz * blocked_size;
  cuda_spline->stride.z = blocked_size;

  cuda_spline->gridInv.x = spline->x_grid.delta_inv;
  cuda_spline->gridInv.y = spline->y_grid.delta_inv;
  cuda_spline->gridInv.z = spline->z_grid.delta_inv;

  cuda_spline->dim.x = spline->x_grid.num;
  cuda_spline->dim.y = spline->y_grid.num;
  cuda_spline->dim.z = spline->z_grid.num;

  // was N now blocked_size
  size_t size = Nx * Ny * Nz * blocked_size * sizeof(double);

  cudaMalloc((void**)&(cuda_spline->coefs), size);
  double* spline_buff = (double*)malloc(size);
  if (!spline_buff)
  {
    fprintf(stderr, "Failed to allocate memory for temporary spline buffer.\n");
    abort();
  }

  // I think we may now be layed out the same so this is not necessary
  for (int ix = 0; ix < Nx; ix++)
    for (int iy = 0; iy < Ny; iy++)
      for (int iz = 0; iz < Nz; iz++)
        for (int isp = 0; isp < blocked_size; isp++)
        {
          if (isp < spline->num_splines)
          {
            spline_buff[ix * cuda_spline->stride.x + iy * cuda_spline->stride.y +
                        iz * cuda_spline->stride.z + isp] =
                spline->coefs[ix * spline->x_stride + iy * spline->y_stride +
                              iz * spline->z_stride + isp];
          }
          else
            spline_buff[ix * cuda_spline->stride.x + iy * cuda_spline->stride.y +
                        iz * cuda_spline->stride.z + isp] = std::nan("nocoef");
        }
  cudaMemcpy(cuda_spline->coefs, (void*)spline_buff, size, cudaMemcpyHostToDevice);

  free(spline_buff);

  return cuda_spline;
}


// extern "C" multi_UBspline_3d_z<Devices::CUDA>*
// create_multi_UBspline_3d_z_cuda (multi_UBspline_3d_z<Devices::CPU>* spline)
// {
//   double B_h[48] = { -1.0/6.0,  3.0/6.0, -3.0/6.0, 1.0/6.0,
// 		     3.0/6.0, -6.0/6.0,  0.0/6.0, 4.0/6.0,
// 		    -3.0/6.0,  3.0/6.0,  3.0/6.0, 1.0/6.0,
// 		     1.0/6.0,  0.0/6.0,  0.0/6.0, 0.0/6.0,
// 		         0.0,     -0.5,      1.0,    -0.5,
// 		         0.0,      1.5,     -2.0,     0.0,
// 		         0.0,     -1.5,      1.0,     0.5,
// 		         0.0,      0.5,      0.0,     0.0,
// 		         0.0,      0.0,     -1.0,     1.0,
// 		         0.0,      0.0,      3.0,    -2.0,
// 		         0.0,      0.0,     -3.0,     1.0,
// 		         0.0,      0.0,      1.0,     0.0 };

//   if (gpu.device_group_size>1)
//   {
//     for(unsigned int i=0; i<gpu.device_group_size; i++)
//     {
//       cudaSetDevice(gpu.device_group_numbers[i]);
//       cudaMemcpyToSymbol(Bcuda, B_h, 48*sizeof(double), 0, cudaMemcpyHostToDevice);
//     }
//     cudaSetDevice(gpu.device_group_numbers[gpu.relative_rank%gpu.device_group_size]);
//   } else
//     cudaMemcpyToSymbol(Bcuda, B_h, 48*sizeof(double), 0, cudaMemcpyHostToDevice);

//   multi_UBspline_3d_z<Devices::CUDA> *cuda_spline =
//     (multi_UBspline_3d_z<Devices::CUDA>*) malloc (sizeof (multi_UBspline_3d_z<Devices::CUDA>));

//   cuda_spline->num_splines = spline->num_splines;
//   cuda_spline->num_split_splines = spline->num_splines;

//   int Nx = spline->x_grid.num+3;
//   int Ny = spline->y_grid.num+3;
//   int Nz = spline->z_grid.num+3;

//   int N = spline->num_splines;
//   int spline_start = 0;
//   if (gpu.device_group_size>1)
//   {
//     if(N<gpu.device_group_size)
//     {
//       if(gpu.rank==0)
//         fprintf(stdout, "Error: Not enough splines (%i) to split across %i devices.\n",N,gpu.device_group_size);
//       abort();
//     }
//     N += gpu.device_group_size-1;
//     N /= gpu.device_group_size;
//     spline_start = N * (gpu.relative_rank%gpu.device_group_size);
// #ifdef SPLIT_SPLINE_DEBUG
//     fprintf (stderr, "splines %i - %i of %i\n",spline_start,spline_start+N,spline->num_splines);
// #endif
//   }
// #ifdef SPLIT_SPLINE_DEBUG
//   else
//     fprintf (stderr, "splines N = %i\n",N);
// #endif
//   cuda_spline->num_split_splines = N;
//   int num_splines = N;
//   if ((N%COALLESCED_SIZE) != 0)
//     N += COALLESCED_SIZE - (N%COALLESCED_SIZE);

//   cuda_spline->stride.x = Ny*Nz*N;
//   cuda_spline->stride.y = Nz*N;
//   cuda_spline->stride.z = N;

//   cuda_spline->gridInv.x = spline->x_grid.delta_inv;
//   cuda_spline->gridInv.y = spline->y_grid.delta_inv;
//   cuda_spline->gridInv.z = spline->z_grid.delta_inv;

//   cuda_spline->dim.x = spline->x_grid.num;
//   cuda_spline->dim.y = spline->y_grid.num;
//   cuda_spline->dim.z = spline->z_grid.num;

//   size_t size = Nx*Ny*Nz*N*sizeof(std::complex<double>);

//   cuda_spline->coefs = (complex_double *) gpu.cuda_memory_manager.allocate(size, "SPO_multi_UBspline_3d_z_cuda");
//   /*
//   cudaMalloc((void**)&(cuda_spline->coefs), size);
//   cudaError_t err = cudaGetLastError();
//   if (err != cudaSuccess) {
//     fprintf (stderr, "Failed to allocate %ld memory for GPU spline coefficients.  Error %s\n",
// 	     size, cudaGetErrorString(err));
//     abort();
//   }*/

//   std::complex<double> *spline_buff = (std::complex<double>*)malloc(size);
//   if (!spline_buff) {
//     fprintf (stderr, "Failed to allocate memory for temporary spline buffer.\n");
//     abort();
//   }

//   for (int ix=0; ix<Nx; ix++)
//     for (int iy=0; iy<Ny; iy++)
//       for (int iz=0; iz<Nz; iz++)
//         for (int isp=0; isp < N; isp++)
//         {
//           int curr_sp=isp+spline_start;
//           if ((curr_sp<spline->num_splines) && (isp<num_splines))
//           {
//             spline_buff[ix*cuda_spline->stride.x +
//                         iy*cuda_spline->stride.y +
//                         iz*cuda_spline->stride.z + isp] =
//               spline->coefs[ix*spline->x_stride +
//                             iy*spline->y_stride +
//                             iz*spline->z_stride + curr_sp];
//           } else
//             spline_buff[ix*cuda_spline->stride.x +
//                         iy*cuda_spline->stride.y +
//                         iz*cuda_spline->stride.z + isp] = 0.0;
//         }
//   cudaMemcpy(cuda_spline->coefs, spline_buff, size, cudaMemcpyHostToDevice);

//   cuda_spline->stride.x = 2*Ny*Nz*N;
//   cuda_spline->stride.y = 2*Nz*N;
//   cuda_spline->stride.z = 2*N;

//   free(spline_buff);

//   return cuda_spline;
// }
