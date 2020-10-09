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



add_library(platform_device INTERFACE)
target_include_directories(platform_device INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")

if(QMC_ENABLE_CUDA)
  target_link_libraries(platform_device INTERFACE CUDA::cudart)
endif()

if(QMC_ENABLE_ROCM)
  target_link_libraries(platform_device INTERFACE ROCM::ROCr)
endif()

subdirs(tests)
