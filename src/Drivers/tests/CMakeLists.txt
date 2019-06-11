#//////////////////////////////////////////////////////////////////////////////////////
#// This file is distributed under the University of Illinois/NCSA Open Source License.
#// See LICENSE file in top directory for details.
#//
#// Copyright (c) 2019 QMCPACK developers.
#//
#// File developed by:
#// Peter Doak, doakpw@ornl.gov, Oak Ridge National Lab
#//
#// File created by:
#// Peter Doak, doakpw@ornl.gov, Oak Ridge National Lab
#////////////////////////////////////////////////////////////////////////////////

SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${QMCPACK_UNIT_TEST_DIR})

SET(SRC_DIR Drivers)
SET(UTEST_EXE test_${SRC_DIR})
SET(UTEST_NAME unit_test_${SRC_DIR})


ADD_EXECUTABLE(${UTEST_EXE} ../../Utilities/catch-main.cpp ../MiniQMCOptions.cpp test_MiniQMCOptions.cpp)
TARGET_LINK_LIBRARIES(${UTEST_EXE} qmcutil ${MPI_LIBRARY})

ADD_UNIT_TEST(${UTEST_NAME} "${QMCPACK_UNIT_TEST_DIR}/${UTEST_EXE}")
