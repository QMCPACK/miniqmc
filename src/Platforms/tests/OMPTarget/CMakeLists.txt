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


set(OMPTARGET_SRC_NAMES
  math
  reduction
  target_task
  complex
  complex_reduction
  math_with_complex_header
  memory_interop
  partial_update
  partial_update_full
  map_always
  icpx_opencl_wrong_number
  nowait_taskwait
  taskloop
  runtime_mem
  map_transfer
  allocator
  vector)

if(OMP_TESTS_ALL_IN_ONE)
  set(OMPTARGET_SRCS)
  foreach(NAME IN LISTS OMPTARGET_SRC_NAMES)
    list(APPEND OMPTARGET_SRCS test_omp_${NAME}.cpp)
  endforeach()
  set(NAME all_in_one)
  set(UTEST_EXE test_omptarget_${NAME})
  set(UTEST_NAME unit_${UTEST_EXE})
  add_executable(${UTEST_EXE} ${OMPTARGET_SRCS})
  target_link_libraries(${UTEST_EXE} platform_runtime catch_main)
  add_unit_test(${UTEST_NAME} 1 1 $<TARGET_FILE:${UTEST_EXE}>)
else()
  foreach(
    NAME IN LISTS OMPTARGET_SRC_NAMES)
    set(UTEST_EXE test_omptarget_${NAME})
    set(UTEST_NAME unit_${UTEST_EXE})
    add_executable(${UTEST_EXE} test_omp_${NAME}.cpp)
    target_link_libraries(${UTEST_EXE} platform_runtime catch_main)
    add_unit_test(${UTEST_NAME} 1 1 $<TARGET_FILE:${UTEST_EXE}>)
  endforeach()
endif()

set(UTEST_EXE test_omptarget_blas)
set(UTEST_NAME unit_${UTEST_EXE})

add_executable(${UTEST_EXE} test_omp_BLAS.cpp)
target_link_libraries(${UTEST_EXE} catch_main platform_LA)

if(USE_OBJECT_TARGET)
  target_link_libraries(${UTEST_EXE} platform_omptarget_LA)
endif()

add_unit_test(${UTEST_NAME} 1 1 $<TARGET_FILE:${UTEST_EXE}>)
