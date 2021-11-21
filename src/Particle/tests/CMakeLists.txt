#//////////////////////////////////////////////////////////////////////////////////////
#// This file is distributed under the University of Illinois/NCSA Open Source License.
#// See LICENSE file in top directory for details.
#//
#// Copyright (c) 2020 QMCPACK developers.
#//
#// File developed by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
#//
#// File created by: Mark Dewing, markdewing@gmail.com, University of Illinois at Urbana-Champaign
#//////////////////////////////////////////////////////////////////////////////////////

set(SRC_DIR particle)
set(UTEST_EXE test_${SRC_DIR})
set(UTEST_NAME unit_test_${SRC_DIR})

add_executable(${UTEST_EXE} test_particle.cpp)
target_link_libraries(${UTEST_EXE} catch_main qmcparticle qmcutil ${QMC_UTIL_LIBS})

add_unit_test(${UTEST_NAME} 1 1 "${UTEST_EXE}")
