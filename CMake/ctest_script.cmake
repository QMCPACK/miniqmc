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
IF ( NOT QMC_SOURCE_DIR )
    SET( QMC_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/.." )
ENDIF()
IF ( NOT MAKE_CMD )
    SET( MAKE_CMD make )
ENDIF()

# Check that we specified the build type to run
IF ( "$ENV{MINIQMC_TEST_SUBMIT_NAME}" )
    SET( CTEST_BUILD_NAME "$ENV{MINIQMC_TEST_SUBMIT_NAME}" )
ELSE()
    SET( CTEST_BUILD_NAME "MINIQMC_TEST_SUBMIT_NAME-unset" )
ENDIF()

# Set the number of processors
IF( NOT DEFINED N_PROCS )
    SET( N_PROCS $ENV{N_PROCS} )
ENDIF()
IF( NOT DEFINED N_PROCS )
    SET(N_PROCS 1)
    # Linux:
    SET(cpuinfo_file "/proc/cpuinfo")
    IF(EXISTS "${cpuinfo_file}")
        FILE(STRINGS "${cpuinfo_file}" procs REGEX "^processor.: [0-9]+$")
        list(LENGTH procs N_PROCS)
    ENDIF()
    # Mac:
    IF(APPLE)
        find_program(cmd_sys_pro "system_profiler")
        if(cmd_sys_pro)
            execute_process(COMMAND ${cmd_sys_pro} OUTPUT_VARIABLE info)
            STRING(REGEX REPLACE "^.*Total Number of Cores: ([0-9]+).*$" "\\1" N_PROCS "${info}")
        ENDIF()
    ENDIF()
    # Windows:
    IF(WIN32)
        SET(N_PROCS "$ENV{NUMBER_OF_PROCESSORS}")
    ENDIF()
ENDIF()

MESSAGE("Testing with ${N_PROCS} processors")

# Set the number of processors for compilation and running tests
IF( NOT DEFINED N_PROCS_BUILD )
    SET( N_PROCS_BUILD $ENV{N_PROCS_BUILD} )
ENDIF()
IF( NOT DEFINED N_PROCS_BUILD )
    SET( N_PROCS_BUILD ${N_PROCS} )
ENDIF()
IF( NOT DEFINED ENV{N_CONCURRENT_TESTS} )
    SET( N_CONCURRENT_TESTS ${N_PROCS} )
ENDIF()

# Set basic variables
SET( CTEST_PROJECT_NAME "miniQMC" )
SET( CTEST_SOURCE_DIRECTORY "${QMC_SOURCE_DIR}" )
SET( CTEST_BINARY_DIRECTORY "." )
SET( CTEST_DASHBOARD "Nightly" )
SET( CTEST_TEST_TIMEOUT 900 )
SET( CTEST_CUSTOM_MAXIMUM_NUMBER_OF_ERRORS 500 )
SET( CTEST_CUSTOM_MAXIMUM_NUMBER_OF_WARNINGS 500 )
SET( CTEST_CUSTOM_MAXIMUM_PASSED_TEST_OUTPUT_SIZE 100000 )
SET( CTEST_CUSTOM_MAXIMUM_FAILED_TEST_OUTPUT_SIZE 100000 )
SET( NIGHTLY_START_TIME "18:00:00 EST" )
SET( CTEST_NIGHTLY_START_TIME "22:00:00 EST" )
SET( CTEST_COMMAND "\"${CTEST_EXECUTABLE_NAME}\" -D ${CTEST_DASHBOARD}" )
IF ( BUILD_SERIAL )
    SET( CTEST_BUILD_COMMAND "${MAKE_CMD} -i" )
ELSE ( BUILD_SERIAL )
    SET( CTEST_BUILD_COMMAND "${MAKE_CMD} -i -j ${N_PROCS_BUILD}" )
    MESSAGE("Building with ${N_PROCS_BUILD} processors")
ENDIF( BUILD_SERIAL )

# Clear the binary directory and create an initial cache
CTEST_EMPTY_BINARY_DIRECTORY( ${CTEST_BINARY_DIRECTORY} )
FILE(WRITE "${CTEST_BINARY_DIRECTORY}/CMakeCache.txt" "CTEST_TEST_CTEST:BOOL=1")

MESSAGE("Configure options:")
MESSAGE("   ${CTEST_OPTIONS}")

SET( CTEST_CMAKE_GENERATOR "Unix Makefiles")

IF ( $ENV{TEST_SITE_NAME} )
    SET( CTEST_SITE $ENV{TEST_SITE_NAME} )
ELSE()
    SITE_NAME( HOSTNAME )
    SET( CTEST_SITE ${HOSTNAME} )
ENDIF()

# Configure and run the tests
CTEST_START("${CTEST_DASHBOARD}")
CTEST_UPDATE()
CTEST_CONFIGURE(
    BUILD   ${CTEST_BINARY_DIRECTORY}
    SOURCE  ${CTEST_SOURCE_DIRECTORY}
    OPTIONS "${CTEST_OPTIONS}"
)


# Run the configure, build and tests
CTEST_BUILD()

# run and submit unclassified tests to the default track
CTEST_START( "${CTEST_DASHBOARD}" TRACK "${CTEST_DASHBOARD}" APPEND)
CTEST_TEST( EXCLUDE_LABEL "performance" PARALLEL_LEVEL ${N_CONCURRENT_TESTS} )
CTEST_SUBMIT( PARTS Test )

# Submit the results to oblivion
SET( CTEST_DROP_METHOD "https" )
SET( CTEST_DROP_SITE "cdash.qmcpack.org" )
SET( CTEST_DROP_LOCATION "/CDash/submit.php?project=miniQMC" )
SET( CTEST_DROP_SITE_CDASH TRUE )
SET( DROP_SITE_CDASH TRUE )
CTEST_SUBMIT( PARTS Configure Build Test )

# run and submit the classified tests to their corresponding track
CTEST_START( "${CTEST_DASHBOARD}" TRACK "Performance" APPEND)
CTEST_TEST( INCLUDE_LABEL "performance" PARALLEL_LEVEL 16 )
CTEST_SUBMIT( PARTS Test )
