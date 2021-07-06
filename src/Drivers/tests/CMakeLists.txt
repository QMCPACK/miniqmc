#//////////////////////////////////////////////////////////////////////////////////////
#// This file is distributed under the University of Illinois/NCSA Open Source License.
#// See LICENSE file in top directory for details.
#//
#// Copyright (c) 2020 QMCPACK developers.
#//
#// File developed by:
#// Peter Doak, doakpw@ornl.gov, Oak Ridge National Lab
#// Ye Luo, yeluo@anl.gov, Argonne National Laboratory
#//
#// File created by:
#// Peter Doak, doakpw@ornl.gov, Oak Ridge National Lab
#////////////////////////////////////////////////////////////////////////////////

SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${QMCPACK_UNIT_TEST_DIR})

SET(SRC_DIR Drivers)
SET(UTEST_EXE test_${SRC_DIR})
SET(UTEST_NAME unit_test_${SRC_DIR})

ADD_EXECUTABLE(${UTEST_EXE} ../MiniQMCOptions.cpp test_MiniQMCOptions.cpp)
TARGET_LINK_LIBRARIES(${UTEST_EXE} catch_main qmcutil)

ADD_UNIT_TEST(${UTEST_NAME} 1 1 "${QMCPACK_UNIT_TEST_DIR}/${UTEST_EXE}")

SET(UTEST_EXE test_assignDevice)
ADD_EXECUTABLE(${UTEST_EXE} test_assignDevice.cpp)
TARGET_LINK_LIBRARIES(${UTEST_EXE} catch_main platform_device qmcutil)

ADD_UNIT_TEST(unit_test_assignDevice 1 1 "${QMCPACK_UNIT_TEST_DIR}/${UTEST_EXE}")
ADD_UNIT_TEST(unit_test_assignDevice_mpi 3 1 "${QMCPACK_UNIT_TEST_DIR}/${UTEST_EXE}")

RUN_APP(miniqmc-g111-r1-t16 miniqmc 1 16 miniqmc TEST_ADDED)
RUN_APP(miniqmc_sync_move-g111-r1-t16 miniqmc_sync_move 1 16 miniqmc TEST_ADDED)
RUN_APP(check_spo-g111-r1-t16 check_spo 1 16 check TEST_ADDED)
RUN_APP(check_spo_batched-g111-r1-t16 check_spo 1 16 check TEST_ADDED)
RUN_APP(check_wfc-g111-r1-t16 check_wfc 1 16 check TEST_ADDED)
