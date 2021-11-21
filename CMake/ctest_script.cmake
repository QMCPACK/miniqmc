# ctest script for building, running, and submitting the test results
# Usage:  ctest -s script,build
#   build = debug / optimized / valgrind / coverage
# Note: this test will use use the number of processors defined in the variable N_PROCS,
#   the environment variables
#   N_PROCS, or the number of processors available (if not specified)
#   N_PROCS_BUILD, or N_PROCS (if not specified)
#   N_CONCURRENT_TESTS, or N_PROCS (if not specified)
#   TEST_SITE_NAME, or HOSTNAME (if not specified)

# Get the source directory based on the current directory
if(NOT QMC_SOURCE_DIR)
  set(QMC_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/..")
endif()
if(NOT MAKE_CMD)
  set(MAKE_CMD make)
endif()

# Check that we specified the build type to run
if(DEFINED ENV{MINIQMC_TEST_SUBMIT_NAME})
  set(CTEST_BUILD_NAME "$ENV{MINIQMC_TEST_SUBMIT_NAME}")
else()
  set(CTEST_BUILD_NAME "MINIQMC_TEST_SUBMIT_NAME-unset")
endif()

# Set the number of processors
if(DEFINED ENV{N_PROCS})
  set(N_PROCS $ENV{N_PROCS})
else()
  set(N_PROCS 1)
  # Linux:
  set(cpuinfo_file "/proc/cpuinfo")
  if(EXISTS "${cpuinfo_file}")
    file(STRINGS "${cpuinfo_file}" procs REGEX "^processor.: [0-9]+$")
    list(LENGTH procs N_PROCS)
  endif()
  # Mac:
  if(APPLE)
    find_program(cmd_sys_pro "system_profiler")
    if(cmd_sys_pro)
      execute_process(COMMAND ${cmd_sys_pro} OUTPUT_VARIABLE info)
      string(REGEX REPLACE "^.*Total Number of Cores: ([0-9]+).*$" "\\1" N_PROCS "${info}")
    endif()
  endif()
  # Windows:
  if(WIN32)
    set(N_PROCS "$ENV{NUMBER_OF_PROCESSORS}")
  endif()
endif()

message("Testing with ${N_PROCS} processors")

# Set the number of processors for compilation and running tests
if(DEFINED ENV{N_PROCS_BUILD})
  set(N_PROCS_BUILD $ENV{N_PROCS_BUILD})
else()
  set(N_PROCS_BUILD ${N_PROCS})
endif()

if(DEFINED ENV{N_CONCURRENT_TESTS})
  set(N_CONCURRENT_TESTS $ENV{N_CONCURRENT_TESTS})
else()
  set(N_CONCURRENT_TESTS ${N_PROCS})
endif()

# Set basic variables
set(CTEST_PROJECT_NAME "miniQMC")
set(CTEST_SOURCE_DIRECTORY "${QMC_SOURCE_DIR}")
set(CTEST_BINARY_DIRECTORY ".")
set(CTEST_DASHBOARD "Nightly")
set(CTEST_TEST_TIMEOUT 900)
set(CTEST_CUSTOM_MAXIMUM_NUMBER_OF_ERRORS 500)
set(CTEST_CUSTOM_MAXIMUM_NUMBER_OF_WARNINGS 500)
set(CTEST_CUSTOM_MAXIMUM_PASSED_TEST_OUTPUT_SIZE 100000)
set(CTEST_CUSTOM_MAXIMUM_FAILED_TEST_OUTPUT_SIZE 100000)
set(NIGHTLY_START_TIME "18:00:00 EST")
set(CTEST_NIGHTLY_START_TIME "22:00:00 EST")
set(CTEST_COMMAND "\"${CTEST_EXECUTABLE_NAME}\" -D ${CTEST_DASHBOARD}")
set(CTEST_USE_LAUNCHERS TRUE)

if(BUILD_SERIAL)
  set(CTEST_BUILD_COMMAND "${MAKE_CMD} -i")
else(BUILD_SERIAL)
  set(CTEST_BUILD_COMMAND "${MAKE_CMD} -i -j ${N_PROCS_BUILD}")
  message("Building with ${N_PROCS_BUILD} processors")
endif(BUILD_SERIAL)

# Clear the binary directory and create an initial cache
ctest_empty_binary_directory(${CTEST_BINARY_DIRECTORY})
file(WRITE "${CTEST_BINARY_DIRECTORY}/CMakeCache.txt" "CTEST_TEST_CTEST:BOOL=1")

message("Configure options: ${CMAKE_CONFIGURE_OPTIONS}")

set(CTEST_CMAKE_GENERATOR "Unix Makefiles")

if(DEFINED ENV{TEST_SITE_NAME})
  set(CTEST_SITE $ENV{TEST_SITE_NAME})
else()
  site_name(HOSTNAME)
  set(CTEST_SITE ${HOSTNAME})
endif()

# Run the configure
ctest_start("${CTEST_DASHBOARD}")
ctest_update()
ctest_configure(
  BUILD ${CTEST_BINARY_DIRECTORY}
  SOURCE ${CTEST_SOURCE_DIRECTORY}
  OPTIONS "${CMAKE_CONFIGURE_OPTIONS}")

# Run the build
ctest_build()

# Run and submit unclassified tests to the default track
ctest_test(EXCLUDE_LABEL "performance" PARALLEL_LEVEL ${N_CONCURRENT_TESTS})
ctest_submit(PARTS Test)

# Submit the results to oblivion
set(CTEST_DROP_METHOD "https")
set(CTEST_DROP_SITE "cdash.qmcpack.org")
set(CTEST_DROP_LOCATION "/CDash/submit.php?project=miniQMC")
set(CTEST_DROP_SITE_CDASH TRUE)
set(DROP_SITE_CDASH TRUE)
ctest_submit(PARTS Configure Build Test)

# Run and submit the classified tests to their corresponding track
ctest_start("${CTEST_DASHBOARD}" TRACK "Performance" APPEND)
ctest_test(INCLUDE_LABEL "performance" PARALLEL_LEVEL 16)
ctest_submit(PARTS Test)
