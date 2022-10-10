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

# Create symlinks for a list of files.
function(SYMLINK_LIST_OF_FILES FILENAMES DST_DIR)
  foreach(F IN LISTS FILENAMES)
    get_filename_component(NAME_ONLY ${F} NAME)
    file(CREATE_LINK ${F} "${DST_DIR}/${NAME_ONLY}" SYMBOLIC)
  endforeach()
endfunction()

# Function to copy a directory using symlinks for the files to save storage space.
# Subdirectories are ignored.
# SRC_DIR must be an absolute path
# The -s flag copies using symlinks
# The -t ${DST_DIR} ensures the destination must be a directory
function(COPY_DIRECTORY_USING_SYMLINK SRC_DIR DST_DIR)
  file(MAKE_DIRECTORY "${DST_DIR}")
  # Find all the files but not subdirectories
  file(
    GLOB FILE_ONLY_NAMES
    LIST_DIRECTORIES TRUE
    "${SRC_DIR}/*")
  symlink_list_of_files("${FILE_ONLY_NAMES}" "${DST_DIR}")
endfunction()

# Copy selected files only. h5, pseudopotentials, wavefunction, structure and the used one input file are copied.
function(COPY_DIRECTORY_USING_SYMLINK_LIMITED SRC_DIR DST_DIR ${ARGN})
  file(MAKE_DIRECTORY "${DST_DIR}")
  # Find all the files but not subdirectories
  file(
    GLOB FILE_FOLDER_NAMES
    LIST_DIRECTORIES TRUE
    "${SRC_DIR}/qmc_ref"
    "${SRC_DIR}/qmc-ref"
    "${SRC_DIR}/*.h5"
    "${SRC_DIR}/*.opt.xml"
    "${SRC_DIR}/*.ncpp.xml"
    "${SRC_DIR}/*.BFD.xml"
    "${SRC_DIR}/*.ccECP.xml"
    "${SRC_DIR}/*.py"
    "${SRC_DIR}/*.sh"
    "${SRC_DIR}/*.restart.xml"
    "${SRC_DIR}/Li.xml"
    "${SRC_DIR}/H.xml"
    "${SRC_DIR}/*.L2_test.xml"
    "${SRC_DIR}/*.opt_L2.xml"
    "${SRC_DIR}/*.wfnoj.xml"
    "${SRC_DIR}/*.wfj*.xml"
    "${SRC_DIR}/*.wfs*.xml"
    "${SRC_DIR}/*.wfn*.xml"
    "${SRC_DIR}/*.cuspInfo.xml"
    "${SRC_DIR}/*.H*.xml"
    "${SRC_DIR}/*.structure.xml"
    "${SRC_DIR}/*ptcl.xml")
  symlink_list_of_files("${FILE_FOLDER_NAMES}" "${DST_DIR}")
  list(TRANSFORM ARGN PREPEND "${SRC_DIR}/")
  symlink_list_of_files("${ARGN}" "${DST_DIR}")
endfunction()

# Control copy vs. symlink with top-level variable
function(COPY_DIRECTORY_MAYBE_USING_SYMLINK SRC_DIR DST_DIR ${ARGN})
  if(QMC_SYMLINK_TEST_FILES)
    copy_directory_using_symlink_limited("${SRC_DIR}" "${DST_DIR}" ${ARGN})
  else()
    copy_directory("${SRC_DIR}" "${DST_DIR}")
  endif()
endfunction()

# Symlink or copy an individual file
function(MAYBE_SYMLINK SRC_DIR DST_DIR)
  if(QMC_SYMLINK_TEST_FILES)
    file(CREATE_LINK ${SRC_DIR} ${DST_DIR} SYMBOLIC)
  else()
    file(COPY ${SRC_DIR} DESTINATION ${DST_DIR})
  endif()
endfunction()

# Macro to create the test name
macro(CREATE_TEST_NAME TEST ${ARGN})
  set(TESTNAME "${TEST}")
  foreach(tmp ${ARGN})
    set(TESTNAME "${TESTNAME}--${tmp}")
  endforeach()
  # STRING(REGEX REPLACE "--" "-" TESTNAME ${TESTNAME} )
endmacro()

# Runs qmcpack
#  Note that TEST_ADDED is an output variable
function(
  RUN_APP
  TESTNAME
  APP_NAME
  PROCS
  THREADS
  TEST_LABELS
  TEST_ADDED
  ${ARGN})
  math(EXPR TOT_PROCS "${PROCS} * ${THREADS}")
  set(TEST_ADDED_TEMP FALSE)
  if(HAVE_MPI)
    if(${TOT_PROCS} GREATER ${TEST_MAX_PROCS})
      message(VERBOSE "Disabling test ${TESTNAME} (exceeds maximum number of processors ${TEST_MAX_PROCS})")
    else()
      add_test(NAME ${TESTNAME} COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${PROCS} ${MPIEXEC_PREFLAGS}
                                        ${APP_NAME} ${ARGN})
      set_tests_properties(
        ${TESTNAME}
        PROPERTIES FAIL_REGULAR_EXPRESSION
                   "ERROR"
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
      add_test(NAME ${TESTNAME} COMMAND ${APP_NAME} ${ARGN})
      set_tests_properties(
        ${TESTNAME}
        PROPERTIES FAIL_REGULAR_EXPRESSION
                   "ERROR"
                   PROCESSORS
                   ${TOT_PROCS}
                   PROCESSOR_AFFINITY
                   TRUE
                   ENVIRONMENT
                   OMP_NUM_THREADS=${THREADS})
      set(TEST_ADDED_TEMP TRUE)
    else()
      message(VERBOSE "Disabling test ${TESTNAME} (building without MPI)")
    endif()
  endif()

  if(TEST_ADDED_TEMP)
    if(QMC_ENABLE_CUDA OR QMC_ENABLE_ROCM OR ENABLE_OFFLOAD)
      set_tests_properties(${TESTNAME} PROPERTIES RESOURCE_LOCK exclusively_owned_gpus)
    endif()

    if(ENABLE_OFFLOAD)
      set_property(
        TEST ${TESTNAME}
        APPEND
        PROPERTY ENVIRONMENT "OMP_TARGET_OFFLOAD=mandatory")
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
