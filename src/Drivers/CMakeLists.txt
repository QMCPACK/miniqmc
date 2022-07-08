#//////////////////////////////////////////////////////////////////////////////////////
#// This file is distributed under the University of Illinois/NCSA Open Source License.
#// See LICENSE file in top directory for details.
#//
#// Copyright (c) 2019 QMCPACK developers.
#//
#// File developed by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
#//
#// File created by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
#//////////////////////////////////////////////////////////////////////////////////////

set(DRIVERS check_spo check_spo_batched check_spo_batched_reduction check_wfc miniqmc miniqmc_sync_move)

foreach(p ${DRIVERS})
  add_executable(${p} ${p}.cpp)
  target_link_libraries(${p} qmcwfs qmcparticle qmcutil platform_runtime ${QMC_UTIL_LIBS})
  set_target_properties(${p} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${miniqmc_BINARY_DIR}/bin)
  install(TARGETS ${p} RUNTIME DESTINATION bin)
endforeach(p)

subdirs(tests)
