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


set(SRCS OMPallocator.hpp  OMPMatrix.h  OMPstd.hpp  OMPTinyVector.h  OMPVector.h)
add_library(platform_omptarget INTERFACE ${OMP_SRCS})
target_include_directories(platform_omptarget INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")

if(ENABLE_OFFLOAD)
    subdirs(tests)
endif()
