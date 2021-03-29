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


set(DEVICE_SRCS
    MemoryUsage.cpp
    Host/sysutil.cpp
)

add_library(platform_device ${DEVICE_SRCS})
target_include_directories(platform_device PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

SUBDIRS(OMPTarget)
target_link_libraries(platform_device PUBLIC platform_omptarget)

if(QMC_ENABLE_CUDA)
  SUBDIRS(CUDA)
  target_link_libraries(platform_device PUBLIC platform_cuda)
endif()

if(QMC_ENABLE_ROCM)
  target_link_libraries(platform_device INTERFACE ROCM::HIP)
endif()

if(QMC_ENABLE_ONEAPI)
  target_link_libraries(platform_device INTERFACE ONEAPI::DPCPP)
endif()

subdirs(tests)
