#//////////////////////////////////////////////////////////////////////////////////////
#// This file is distributed under the University of Illinois/NCSA Open Source License.
#// See LICENSE file in top directory for details.
#//
#// Copyright (c) 2020 QMCPACK developers.
#//
#// File developed by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
#//
#// File created by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
#//////////////////////////////////////////////////////////////////////////////////////


SET(CUDA_SRCS
  CUDAfill.cpp
  CUDAallocator.cpp
  CUDAruntime.cpp
)

add_library(platform_cuda ${CUDA_SRCS})
TARGET_LINK_LIBRARIES(platform_cuda PUBLIC CUDA::cudart)
