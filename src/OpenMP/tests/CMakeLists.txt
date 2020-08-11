#//////////////////////////////////////////////////////////////////////////////////////
#// This file is distributed under the University of Illinois/NCSA Open Source License.
#// See LICENSE file in top directory for details.
#//
#// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
#//
#// File developed by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
#//
#// File created by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
#//////////////////////////////////////////////////////////////////////////////////////

SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${QMCPACK_UNIT_TEST_DIR})

SET(UTEST_EXE test_omptarget_math)
SET(UTEST_NAME unit_${UTEST_EXE})
ADD_EXECUTABLE(${UTEST_EXE} test_omp_math.cpp)
TARGET_LINK_LIBRARIES(${UTEST_EXE} catch_main ${MPI_LIBRARY})
ADD_UNIT_TEST(${UTEST_NAME} "${QMCPACK_UNIT_TEST_DIR}/${UTEST_EXE}")

SET(UTEST_EXE test_omptarget_reduction)
SET(UTEST_NAME unit_${UTEST_EXE})
ADD_EXECUTABLE(${UTEST_EXE} test_omp_reduction.cpp)
TARGET_LINK_LIBRARIES(${UTEST_EXE} catch_main ${MPI_LIBRARY})
ADD_UNIT_TEST(${UTEST_NAME} "${QMCPACK_UNIT_TEST_DIR}/${UTEST_EXE}")

SET(UTEST_EXE test_omptarget_target_task)
SET(UTEST_NAME unit_${UTEST_EXE})
ADD_EXECUTABLE(${UTEST_EXE} test_omp_target_task.cpp)
TARGET_LINK_LIBRARIES(${UTEST_EXE} catch_main ${MPI_LIBRARY})
ADD_UNIT_TEST(${UTEST_NAME} "${QMCPACK_UNIT_TEST_DIR}/${UTEST_EXE}")
