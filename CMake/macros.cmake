#############################################################
# Functions for adding tests to ctest
#############################################################
# Useful macros to run an existing executable:
#   RUN_QMC_APP
#     Run QMCPACK with a given number of threads and MPI processes
#
#   QMC_RUN_AND_CHECK
#     Run QMCPACK and check scalar output values.  This is the
#     primary function used for system tests.
#
#   SIMPLE_RUN_AND_CHECK
#     Run QMCPACK on the given input file and check output
#     using a specified script
#############################################################

# Function to copy a directory
function(COPY_DIRECTORY SRC_DIR DST_DIR)
  execute_process(COMMAND ${CMAKE_COMMAND} -E copy_directory "${SRC_DIR}" "${DST_DIR}")
endfunction()

# Function to copy a directory using symlinks for the files. This saves storage
# space with large test files.
# SRC_DIR must be an absolute path
# The -s flag copies using symlinks
# The -T ${DST_DIR} ensures the destination is copied as the directory, and not
#  placed as a subdirectory if the destination already exists.
function(COPY_DIRECTORY_USING_SYMLINK SRC_DIR DST_DIR)
  execute_process(COMMAND cp -as --remove-destination "${SRC_DIR}" -T "${DST_DIR}")
endfunction()

# Copy files, but symlink the *.h5 files (which are the large ones)
function(COPY_DIRECTORY_SYMLINK_H5 SRC_DIR DST_DIR)
  # Copy everything but *.h5 files and pseudopotential files
  file(
    COPY "${SRC_DIR}/"
    DESTINATION "${DST_DIR}"
    PATTERN "*.h5" EXCLUDE
    PATTERN "*.opt.xml" EXCLUDE
    PATTERN "*.ncpp.xml" EXCLUDE
    PATTERN "*.BFD.xml" EXCLUDE)

  # Now find and symlink the *.h5 files and psuedopotential files
  file(GLOB_RECURSE H5 "${SRC_DIR}/*.h5" "${SRC_DIR}/*.opt.xml" "${SRC_DIR}/*.ncpp.xml" "${SRC_DIR}/*.BFD.xml")
  foreach(F IN LISTS H5)
    file(RELATIVE_PATH R "${SRC_DIR}" "${F}")
    #MESSAGE("Creating symlink from  ${SRC_DIR}/${R} to ${DST_DIR}/${R}")
    execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink "${SRC_DIR}/${R}" "${DST_DIR}/${R}")
  endforeach()
endfunction()

# Control copy vs. symlink with top-level variable
function(COPY_DIRECTORY_MAYBE_USING_SYMLINK SRC_DIR DST_DIR)
  if(QMC_SYMLINK_TEST_FILES)
    #COPY_DIRECTORY_USING_SYMLINK("${SRC_DIR}" "${DST_DIR}")
    copy_directory_symlink_h5("${SRC_DIR}" "${DST_DIR}")
  else()
    copy_directory("${SRC_DIR}" "${DST_DIR}")
  endif()
endfunction()

# Symlink or copy an individual file
function(MAYBE_SYMLINK SRC_DIR DST_DIR)
  if(QMC_SYMLINK_TEST_FILES)
    execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink "${SRC_DIR}" "${DST_DIR}")
  else()
    execute_process(COMMAND ${CMAKE_COMMAND} -E copy "${SRC_DIR}" "${DST_DIR}")
  endif()
endfunction()

# Macro to add the dependencies and libraries to an executable
macro(ADD_QMC_EXE_DEP EXE)
  # Add the package dependencies
  target_link_libraries(
    ${EXE}
    qmc
    qmcdriver
    qmcham
    qmcwfs
    qmcbase
    qmcutil
    adios_config)
  foreach(l ${QMC_UTIL_LIBS})
    target_link_libraries(${EXE} ${l})
  endforeach(l ${QMC_UTIL_LIBS})
  if(MPI_LIBRARY)
    target_link_libraries(${EXE} ${MPI_LIBRARY})
  endif(MPI_LIBRARY)
endmacro()

# Macro to create the test name
macro(CREATE_TEST_NAME TEST ${ARGN})
  set(TESTNAME "${TEST}")
  foreach(tmp ${ARGN})
    set(TESTNAME "${TESTNAME}--${tmp}")
  endforeach()
  # STRING(REGEX REPLACE "--" "-" TESTNAME ${TESTNAME} )
endmacro()

# Runs given apps
#  Note that TEST_ADDED is an output variable
function(
  RUN_APP
  TESTNAME
  APPNAME
  PROCS
  THREADS
  TEST_LABELS
  TEST_ADDED
  ${ARGN})
  math(EXPR TOT_PROCS "${PROCS} * ${THREADS}")
  set(APP_EXE $<TARGET_FILE:${APPNAME}>)
  set(TEST_ADDED_TEMP FALSE)
  if(HAVE_MPI)
    if(${TOT_PROCS} GREATER ${TEST_MAX_PROCS})
      message("Disabling test ${TESTNAME} (exceeds maximum number of processors ${TEST_MAX_PROCS})")
    else()
      add_test(NAME ${TESTNAME} COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${PROCS} ${APP_EXE} ${ARGN})
      set_tests_properties(
        ${TESTNAME}
        PROPERTIES FAIL_REGULAR_EXPRESSION
                   "${TEST_FAIL_REGULAR_EXPRESSION}"
                   PROCESSORS
                   ${TOT_PROCS}
                   PROCESSOR_AFFINITY
                   TRUE
                   ENVIRONMENT
                   OMP_NUM_THREADS=${THREADS})
      set(TEST_ADDED_TEMP TRUE)
    endif()
  else()
    if((${PROCS} STREQUAL "1"))
      add_test(NAME ${TESTNAME} COMMAND ${APP_EXE} ${ARGN})
      set_tests_properties(
        ${TESTNAME}
        PROPERTIES FAIL_REGULAR_EXPRESSION
                   "${TEST_FAIL_REGULAR_EXPRESSION}"
                   PROCESSORS
                   ${TOT_PROCS}
                   PROCESSOR_AFFINITY
                   TRUE
                   ENVIRONMENT
                   OMP_NUM_THREADS=${THREADS})
      set(TEST_ADDED_TEMP TRUE)
    else()
      message("Disabling test ${TESTNAME} (building without MPI)")
    endif()
  endif()

  if(TEST_ADDED_TEMP)
    if(ENABLE_OFFLOAD
       OR QMC_ENABLE_CUDA
       OR QMC_ENABLE_ROCM
       OR QMC_ENABLE_ONEAPI)
      set_tests_properties(${TESTNAME} PROPERTIES RESOURCE_LOCK exclusively_owned_gpus)
    endif()
    set_property(
      TEST ${TESTNAME}
      APPEND
      PROPERTY LABELS ${TEST_LABELS})
  endif()

  set(${TEST_ADDED}
      ${TEST_ADDED_TEMP}
      PARENT_SCOPE)
endfunction()
