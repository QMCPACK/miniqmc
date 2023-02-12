#//////////////////////////////////////////////////////////////////////////////////////
#// This file is distributed under the University of Illinois/NCSA Open Source License.
#// See LICENSE file in top directory for details.
#//
#// Copyright (c) 2021 QMCPACK developers.
#//
#// File developed by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
#//
#// File created by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
#//////////////////////////////////////////////////////////////////////////////////////

add_library(catch_main catch_main.cpp)
target_include_directories(catch_main PUBLIC "${PROJECT_SOURCE_DIR}/external_codes/catch")
target_link_libraries(catch_main PRIVATE platform_runtime qmcutil)
if(HAVE_MPI)
  target_compile_definitions(catch_main PRIVATE "CATCH_MAIN_HAVE_MPI")
endif()

####################################
# create libqmcutil
####################################
set(UTILITIES Communicate.cpp NewTimer.cpp TimerManager.cpp XMLWriter.cpp tinyxml/tinyxml2.cpp qmcpack_version.cpp ResourceCollection.cpp)

add_library(qmcutil ${UTILITIES})
target_include_directories(qmcutil PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(
  qmcutil
  PUBLIC platform_runtime
  INTERFACE qmc_mpi_cxx)

if(IS_GIT_PROJECT)
  add_dependencies(qmcutil gitrev)
endif()

subdirs(tests)
