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

set(SRC_DIR Drivers)
set(UTEST_EXE test_${SRC_DIR})
set(UTEST_NAME unit_test_${SRC_DIR})

add_executable(${UTEST_EXE} ../MiniQMCOptions.cpp test_MiniQMCOptions.cpp)
target_link_libraries(${UTEST_EXE} catch_main qmcutil)

add_unit_test(${UTEST_NAME} 1 1 "${UTEST_EXE}")

set(UTEST_EXE test_assignDevice)
add_executable(${UTEST_EXE} test_assignDevice.cpp)
target_link_libraries(${UTEST_EXE} catch_main platform_runtime qmcutil)

add_unit_test(unit_test_assignDevice 1 1 "${UTEST_EXE}")
add_unit_test(unit_test_assignDevice_mpi 3 1 "${UTEST_EXE}")

run_app(miniqmc-g111-r1-t16 $<TARGET_FILE:miniqmc> 1 16 miniqmc TEST_ADDED)
run_app(miniqmc_sync_move-g111-r1-t16 $<TARGET_FILE:miniqmc_sync_move> 1 16 miniqmc TEST_ADDED)
run_app(check_spo-g111-r1-t16 $<TARGET_FILE:check_spo> 1 16 check TEST_ADDED)
run_app(check_spo_batched-g111-r1-t16 $<TARGET_FILE:check_spo_batched> 1 16 check TEST_ADDED)
run_app(check_spo_batched_reduction-g111-r1-t16 $<TARGET_FILE:check_spo_batched_reduction> 1 16 check TEST_ADDED)
run_app(check_wfc-g111-r1-t16 $<TARGET_FILE:check_wfc> 1 16 check TEST_ADDED)
