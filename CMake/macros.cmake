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
FUNCTION( COPY_DIRECTORY SRC_DIR DST_DIR )
    EXECUTE_PROCESS( COMMAND ${CMAKE_COMMAND} -E copy_directory "${SRC_DIR}" "${DST_DIR}" )
ENDFUNCTION()

# Function to copy a directory using symlinks for the files. This saves storage
# space with large test files.
# SRC_DIR must be an absolute path
# The -s flag copies using symlinks
# The -T ${DST_DIR} ensures the destination is copied as the directory, and not
#  placed as a subdirectory if the destination already exists.
FUNCTION( COPY_DIRECTORY_USING_SYMLINK SRC_DIR DST_DIR )
    EXECUTE_PROCESS( COMMAND cp -as --remove-destination "${SRC_DIR}" -T "${DST_DIR}" )
ENDFUNCTION()

# Copy files, but symlink the *.h5 files (which are the large ones)
FUNCTION( COPY_DIRECTORY_SYMLINK_H5 SRC_DIR DST_DIR)
    # Copy everything but *.h5 files and pseudopotential files
    FILE(COPY "${SRC_DIR}/" DESTINATION "${DST_DIR}"
         PATTERN "*.h5" EXCLUDE
         PATTERN "*.opt.xml" EXCLUDE
         PATTERN "*.ncpp.xml" EXCLUDE
         PATTERN "*.BFD.xml" EXCLUDE)

    # Now find and symlink the *.h5 files and psuedopotential files
    FILE(GLOB_RECURSE H5 "${SRC_DIR}/*.h5" "${SRC_DIR}/*.opt.xml" "${SRC_DIR}/*.ncpp.xml" "${SRC_DIR}/*.BFD.xml")
    FOREACH(F IN LISTS H5)
      FILE(RELATIVE_PATH R "${SRC_DIR}" "${F}")
      #MESSAGE("Creating symlink from  ${SRC_DIR}/${R} to ${DST_DIR}/${R}")
      EXECUTE_PROCESS(COMMAND ${CMAKE_COMMAND} -E create_symlink "${SRC_DIR}/${R}" "${DST_DIR}/${R}")
    ENDFOREACH()
ENDFUNCTION()

# Control copy vs. symlink with top-level variable
FUNCTION( COPY_DIRECTORY_MAYBE_USING_SYMLINK SRC_DIR DST_DIR )
  IF (QMC_SYMLINK_TEST_FILES)
    #COPY_DIRECTORY_USING_SYMLINK("${SRC_DIR}" "${DST_DIR}")
    COPY_DIRECTORY_SYMLINK_H5("${SRC_DIR}" "${DST_DIR}" )
  ELSE()
    COPY_DIRECTORY("${SRC_DIR}" "${DST_DIR}")
  ENDIF()
ENDFUNCTION()

# Symlink or copy an individual file
FUNCTION(MAYBE_SYMLINK SRC_DIR DST_DIR)
  IF (QMC_SYMLINK_TEST_FILES)
    EXECUTE_PROCESS(COMMAND ${CMAKE_COMMAND} -E create_symlink "${SRC_DIR}" "${DST_DIR}")
  ELSE()
    EXECUTE_PROCESS(COMMAND ${CMAKE_COMMAND} -E copy "${SRC_DIR}" "${DST_DIR}")
  ENDIF()
ENDFUNCTION()


# Macro to add the dependencies and libraries to an executable
MACRO( ADD_QMC_EXE_DEP EXE )
    # Add the package dependencies
    TARGET_LINK_LIBRARIES(${EXE} qmc qmcdriver qmcham qmcwfs qmcbase qmcutil adios_config)
    FOREACH(l ${QMC_UTIL_LIBS})
        TARGET_LINK_LIBRARIES(${EXE} ${l})
    ENDFOREACH(l ${QMC_UTIL_LIBS})
    IF(MPI_LIBRARY)
        TARGET_LINK_LIBRARIES(${EXE} ${MPI_LIBRARY})
    ENDIF(MPI_LIBRARY)
ENDMACRO()



# Macro to create the test name
MACRO( CREATE_TEST_NAME TEST ${ARGN} )
    SET( TESTNAME "${TEST}" )
    FOREACH( tmp ${ARGN} )
        SET( TESTNAME "${TESTNAME}--${tmp}")
    endforeach()
    # STRING(REGEX REPLACE "--" "-" TESTNAME ${TESTNAME} )
ENDMACRO()


# Runs given apps
#  Note that TEST_ADDED is an output variable
FUNCTION( RUN_APP TESTNAME APPNAME PROCS THREADS TEST_LABELS TEST_ADDED ${ARGN} )
    MATH( EXPR TOT_PROCS "${PROCS} * ${THREADS}" )
    SET( APP_EXE "${miniqmc_BINARY_DIR}/bin/${APPNAME}" )
    SET( TEST_ADDED_TEMP FALSE )
    IF ( USE_MPI )
        IF ( ${TOT_PROCS} GREATER ${TEST_MAX_PROCS} )
            MESSAGE("Disabling test ${TESTNAME} (exceeds maximum number of processors ${TEST_MAX_PROCS})")
        ELSEIF ( USE_MPI )
            ADD_TEST( ${TESTNAME} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${PROCS} ${APP_EXE} ${ARGN} )
            SET_TESTS_PROPERTIES( ${TESTNAME} PROPERTIES FAIL_REGULAR_EXPRESSION "${TEST_FAIL_REGULAR_EXPRESSION}" 
                PROCESSORS ${TOT_PROCS} PROCESSOR_AFFINITY TRUE ENVIRONMENT OMP_NUM_THREADS=${THREADS} )
            SET( TEST_ADDED_TEMP TRUE )
        ENDIF()
    ELSE()
        IF ( ( ${PROCS} STREQUAL "1" ) )
            ADD_TEST( ${TESTNAME} ${APP_EXE} ${ARGN} )
            SET_TESTS_PROPERTIES( ${TESTNAME} PROPERTIES FAIL_REGULAR_EXPRESSION "${TEST_FAIL_REGULAR_EXPRESSION}" 
                PROCESSORS ${TOT_PROCS} PROCESSOR_AFFINITY TRUE ENVIRONMENT OMP_NUM_THREADS=${THREADS} )
            SET( TEST_ADDED_TEMP TRUE )
        ELSE()
            MESSAGE("Disabling test ${TESTNAME} (building without MPI)")
        ENDIF()
    ENDIF()

    IF ( TEST_ADDED_TEMP )
      SET_PROPERTY(TEST ${TESTNAME} APPEND PROPERTY LABELS ${TEST_LABELS})
    ENDIF()

    SET( ${TEST_ADDED} ${TEST_ADDED_TEMP} PARENT_SCOPE )
ENDFUNCTION()
