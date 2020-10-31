######################################################################
# GLOBAL
######################################################################
CMAKE_MINIMUM_REQUIRED(VERSION 3.6.0)
cmake_policy(SET CMP0003 NEW)
#Actually set CXX11 flag for Clang on Apple, even llvm non apple clang
#This is contrary to what the --help-policy says
cmake_policy(SET CMP0025 NEW)

######################################################################
PROJECT(miniqmc CXX)
######################################################################

######################################################################
# Version
######################################################################
SET(QMCPACK_VERSION_MAJOR 3)
SET(QMCPACK_VERSION_MINOR 1)
SET(QMCPACK_VERSION_PATCH 0)
SET(QMCPACK_VERSION "${QMCPACK_VERSION_MAJOR}.${QMCPACK_VERSION_MINOR}.${QMCPACK_VERSION_PATCH}")

######################################################################
# CTest
######################################################################
INCLUDE( "${miniqmc_SOURCE_DIR}/CMake/macros.cmake" )
SET( DROP_METHOD "http" )
SET( DROP_SITE "cdash.qmcpack.org" )
SET( DROP_LOCATION "/CDash/submit.php?project=miniQMC" )
SET( TRIGGER_SITE "" )
SET( DROP_SITE_CDASH TRUE )
# Increase timeout per test over the default of 1500 seconds (25 minutes)
SET( DART_TESTING_TIMEOUT 3600 CACHE STRING "Maximum time for one test")
ENABLE_TESTING()
INCLUDE( CTest )

IF ( NOT CMAKE_BUILD_TYPE AND NOT CMAKE_TOOLCHAIN_FILE)
  SET( CMAKE_BUILD_TYPE Release )
ENDIF()
STRING( TOUPPER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE )
MESSAGE( STATUS "CMAKE_BUILD_TYPE is ${CMAKE_BUILD_TYPE}")

######################################################################
# Define the paths for library and executable for out-of-soruce compilation
# Directory where customize cmake files reside
######################################################################
SET (LIBRARY_OUTPUT_PATH ${miniqmc_BINARY_DIR}/lib CACHE PATH "Single output directory for building all libraries.")
SET (EXECUTABLE_OUTPUT_PATH ${miniqmc_BINARY_DIR}/bin CACHE PATH "Single output directory for building all executables.")
SET (PROJECT_CMAKE ${miniqmc_SOURCE_DIR}/CMake)
SET (QMCPACK_UNIT_TEST_DIR ${miniqmc_BINARY_DIR}/tests/bin)

######################################################################
# build include and library paths required for the projects
# For XYZ library, set an environment as
# export XYZ_HOME='root-of-the-library'
# XYZ_HOME/include
# XYZ_HOME/lib
# IF XYZ library is found, XYZ_FOUND is set to true
#
# find_package is used with cmake 2.8 and higher
######################################################################
#if(APPLE)
#  set(QMC_INCLUDE_PATHS ${QMC_INCLUDE_PATHS} /sw/include)
#  set(QMC_LIBRARY_PATHS ${QMC_LIBRARY_PATHS} /sw/lib)
#endif(APPLE)

######################################################################
# Set the compiler-time parameters
# OHMMS_DIM =  dimension of the problem
# OHMMS_INDEXTYPE = type of index
# OHMMS_PRECISION  = base precision, float, double etc
# OHMMS_PRECISION_FULL  = full precision, double etc
# QMC_COMPLEX = true if using complex wavefunctions
# QMC_MPI =  enable MPI
# QMC_OMP = enable OMP
######################################################################
SET(OHMMS_DIM 3 CACHE STRING "Select physical dimension")
SET(OHMMS_INDEXTYPE int)
SET(OHMMS_ORTHO 0)
MESSAGE(STATUS "defining the float point precision")
SET(OHMMS_PRECISION_FULL double)
SET(QMC_MIXED_PRECISION 0 CACHE BOOL "Enable/disable mixed precision")
IF(QMC_MIXED_PRECISION)
  SET(OHMMS_PRECISION float)
  SET(MIXED_PRECISION 1)
ELSE(QMC_MIXED_PRECISION)
  SET(OHMMS_PRECISION double)
ENDIF(QMC_MIXED_PRECISION)
MESSAGE("   Base precision = ${OHMMS_PRECISION}")
MESSAGE("   Full precision = ${OHMMS_PRECISION_FULL}")

# Code coverage
SET(GCOV_SUPPORTED FALSE)
SET(ENABLE_GCOV FALSE CACHE BOOL "Enable code coverage")

######################################################################
# enable MPI and OPNEMP  if possible
######################################################################
SET(QMC_MPI 0 CACHE BOOL "Enable/disable MPI")
SET(QMC_OMP 1 CACHE BOOL "Enable/disable OpenMP")
SET(QMC_COMPLEX 0 CACHE STRING "Build for complex binary")

######################################################################
# Standard test
######################################################################
SET(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake" ${CMAKE_MODULE_PATH})


######################################################################
# add macros to compile einspline if necessary
######################################################################
INCLUDE(${CMAKE_ROOT}/Modules/CheckFunctionExists.cmake)

CHECK_FUNCTION_EXISTS(sincos HAVE_SINCOS)
CHECK_FUNCTION_EXISTS(posix_memalign HAVE_POSIX_MEMALIGN)

######################################################################
# Build options
# QMC_BUILD_STATIC build static/dynamic  libraries
# BUILD_QMCTOOLS   build utility programs
# BUILD_SANDBOX    build test programs
######################################################################
SET(BUILD_UNIT_TESTS 1 CACHE BOOL "Build unit tests")
SET(BUILD_LMYENGINE_INTERFACE 1 CACHE BOOL "Build LMY engine")
iF(MIXED_PRECISION AND BUILD_LMYENGINE_INTERFACE)
  MESSAGE(STATUS "LMY engine is not compatiable with CPU mixed precision build! Disabling LMY engine")
  SET(BUILD_LMYENGINE_INTERFACE 0)
ENDIF()
SET(BUILD_AFQMC 0 CACHE BOOL "Build with AFQMC")
# AFQMC requires MPI
If (BUILD_AFQMC AND NOT QMC_MPI)
  MESSAGE(FATAL_ERROR "AFQMC requires building with MPI (QMC_MPI=1). Set BUILD_AFQMC=0 or configure MPI.")
ENDIF()

SET(BUILD_FCIQMC 0 CACHE BOOL "Build with FCIQMC")
SET(QMC_BUILD_STATIC 0 CACHE BOOL "Link to static libraries")
SET(ENABLE_TIMERS 1 CACHE BOOL "Enable internal timers")

######################################################################
#list of options to determine libraries. First, try to use libraries available
#FindXYZ.cmake will set the values if successful
######################################################################
#SET(ENABLE_PHDF5 0 CACHE BOOL "Enable phdf5 for output")
#SET(HAVE_LIBHDF5 0)

######################################################
# Git information (see src/CMakeLists.txt for the
# command to get repository information)
######################################################
FIND_PACKAGE(Git)

IF (GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
  SET(IS_GIT_PROJECT 1)
ENDIF()

######################################################
#COMPILER choose one of the cmake files to customize the compiler options
#If nothing is chosen, default settings by cmake will  be used.
#If the automatic detection does not work, comment out everything 
#upto COMPILER MANUAL SELECTION and use one of the customized cmake file.
######################################################
INCLUDE(${CMAKE_ROOT}/Modules/CheckCXXCompilerFlag.cmake)


######################################################
# DO NOT MODIFY BELOW
# If you like to customize your build, create a toolchain as instructed
######################################################
######################################################
# now set the compiler flags and the macros in src/config.h
######################################################
IF(CMAKE_TOOLCHAIN_FILE)
  MESSAGE(STATUS "Using ${CMAKE_TOOLCHAIN_FILE} toolchain ")
ELSE(CMAKE_TOOLCHAIN_FILE)
  MESSAGE(STATUS "Trying to figure out .....")
  SET(ENABLE_OPENMP 0)

  ######################################################
  # Try to identify CPU identity
  ######################################################
  SET(CPU_IDENTITY "generic")
  #INCLUDE(${PROJECT_CMAKE}/CheckProcessorID.cmake)

  #------------------------
  # On Cray's machine
  #------------------------
  GET_FILENAME_COMPONENT(BASE_CXX_COMPILER_NAME ${CMAKE_CXX_COMPILER} NAME)
  IF($ENV{CRAYPE_VERSION} MATCHES "." AND NOT BASE_CXX_COMPILER_NAME STREQUAL "CC")
    MESSAGE(WARNING "Cray Programming Environment has been detected but C++ compiler wrapper CC is not used! "
                    "Please use cc/CC as the C/C++ compiler to ensure correct cross-compiling for the compute nodes "
                    "unless you understand cross-compiling and intend not to use Cray compiler wrappers.")
  ENDIF()
  IF(CMAKE_SYSTEM_NAME STREQUAL "CrayLinuxEnvironment")
    MESSAGE(STATUS "Running on a Cray machine.")

    IF(NOT $ENV{CRAYPE_LINK_TYPE} STREQUAL "dynamic")
      MESSAGE(WARNING "Cray Programming Environment uses static linking by default. "
                      "We prefer dynamic linking which makes library searching easier. "
                      "Set environment variable CRAYPE_LINK_TYPE to \"dynamic\" enables dynamic linking.\n")
    ENDIF()

    IF ( NOT DEFINED MPIEXEC_EXECUTABLE )
      FIND_PROGRAM(MPIEXEC_EXECUTABLE aprun)
      IF(NOT MPIEXEC_EXECUTABLE)
        # For slurm use srun
        FIND_PROGRAM(MPIEXEC_EXECUTABLE srun)
      ENDIF(NOT MPIEXEC_EXECUTABLE)
    ENDIF()
  ELSE()
    IF($ENV{CRAYPE_VERSION} MATCHES "." AND BASE_CXX_COMPILER_NAME STREQUAL "CC")
      MESSAGE(FATAL_ERROR "Cray compiler wrapper detected. Empty the build folder and rerun cmake with -DCMAKE_SYSTEM_NAME=CrayLinuxEnvironment added.")
    ENDIF()
  ENDIF()

  #------------------------------------
  # Identify the compiler
  #------------------------------------
  IF ( CMAKE_CXX_COMPILER_WORKS )
    IF( CMAKE_CXX_COMPILER_ID MATCHES "GNU" )
      SET( COMPILER GNU )
    ELSEIF( CMAKE_CXX_COMPILER_ID MATCHES "XL" )
      SET( COMPILER IBM )
    ELSEIF( CMAKE_CXX_COMPILER_ID MATCHES "Intel" )
      SET( COMPILER Intel )
    ELSEIF( CMAKE_CXX_COMPILER_ID MATCHES "PGI" )
      SET( COMPILER PGI )
    ELSEIF( CMAKE_CXX_COMPILER_ID MATCHES "Cray" )
      SET( COMPILER Cray )
    ELSEIF( CMAKE_CXX_COMPILER_ID MATCHES "Clang" )
      SET( COMPILER Clang )
    ELSE()
      SET(USING_DEFAULT TRUE)
      MESSAGE("${CMAKE_CXX_COMPILER_ID}")
      MESSAGE(WARNING "Unknown C++ compiler, default flags will be used")
    ENDIF()
  ELSE()
    MESSAGE(WARNING "No compiler identified")
  ENDIF()
  MESSAGE(STATUS "Compiler: ${COMPILER}")

  #------------------------------------
  # Include compiler-specific cmake file
  #------------------------------------
  IF( ${COMPILER} MATCHES "IBM" )
    INCLUDE(${PROJECT_CMAKE}/IBMCompilers.cmake)
  ELSEIF( ${COMPILER} MATCHES "Intel" )
    INCLUDE(${PROJECT_CMAKE}/IntelCompilers.cmake)
  ELSEIF( ${COMPILER} MATCHES "GNU" )
    INCLUDE(${PROJECT_CMAKE}/GNUCompilers.cmake)
  ELSEIF( ${COMPILER} MATCHES "Clang" )
    INCLUDE(${PROJECT_CMAKE}/ClangCompilers.cmake)
  ELSEIF( ${COMPILER} MATCHES "PGI" )
    INCLUDE(${PROJECT_CMAKE}/PGICompilers.cmake)
  ELSE()
    MESSAGE(WARNING "No default file for compiler (${COMPILER})")
  ENDIF()

  # requires C++11 standard
  SET(CMAKE_CXX_STANDARD 11)
  SET(CMAKE_CXX_STANDARD_REQUIRED ON)
  SET(CMAKE_CXX_EXTENSIONS OFF)

ENDIF(CMAKE_TOOLCHAIN_FILE)

IF (ENABLE_GCOV)
  IF (NOT GCOV_SUPPORTED)
    MESSAGE(FATAL_ERROR "ENABLE_GCOV was specified but compiler does not support GCC coverage flag")
  ENDIF()
ENDIF(ENABLE_GCOV)

IF (BUILD_AFQMC AND NOT APPLE)
  LINK_LIBRARIES("rt")
ENDIF()
  
#-------------------------------------------------------------------
#  check MPI installation
#-------------------------------------------------------------------
ADD_LIBRARY(qmc_mpi_cxx INTERFACE)
IF(QMC_MPI)
  # for backward compatibility with MPIEXEC
  IF(DEFINED MPIEXEC AND NOT DEFINED MPIEXEC_EXECUTABLE)
    MESSAGE(WARNING "Setting MPIEXEC_EXECUTABLE based on MPIEXEC. MPIEXEC has been deprecated! Use MPIEXEC_EXECUTABLE instead!")
    SET(MPIEXEC_EXECUTABLE ${MPIEXEC})
  ENDIF(DEFINED MPIEXEC AND NOT DEFINED MPIEXEC_EXECUTABLE)
  SET(MPI_CXX_SKIP_MPICXX TRUE)
  IF(NOT CMAKE_CROSSCOMPILING)
    SET(MPI_DETERMINE_LIBRARY_VERSION TRUE)
  ENDIF()
  FIND_PACKAGE(MPI COMPONENTS CXX)

  IF(NOT MPI_FOUND)
    MESSAGE(FATAL_ERROR "MPI support not found! Provide MPI compiler wrappers or build without MPI by passing '-DQMC_MPI=0' to cmake.")
  ENDIF(NOT MPI_FOUND)

  IF(${MPI_CXX_LIBRARY_VERSION_STRING} MATCHES "MVAPICH2")
    STRING(REGEX REPLACE "\n" ";" ONE_LINE "${MPI_CXX_LIBRARY_VERSION_STRING}")
    STRING(REGEX REPLACE " +|\t" ";" ONE_LINE "${ONE_LINE}")
    LIST (GET ONE_LINE 3 MVAPICH2_VERSION)
    MESSAGE(STATUS "MVAPICH2 version ${MVAPICH2_VERSION}")
    IF(${MVAPICH2_VERSION} VERSION_LESS "2.4" AND NOT ${MPI_CXX_LIBRARY_VERSION_STRING} MATCHES "disable-registration-cache")
        MESSAGE(FATAL_ERROR "MVAPICH2 (version < 2.4) with registration cache enabled breaks QMCPACK. "
                            "Use a different MPI library or a MVAPICH2 >=2.4 "
                            "or build MVAPICH2 with --disable-registration-cache configure option.")
    ENDIF()
  ENDIF()

  IF(${MPI_CXX_LIBRARY_VERSION_STRING} MATCHES "Open MPI")
    IF(NOT MPIEXEC_PREFLAGS)
      SET(MPIEXEC_PREFLAGS "--bind-to;none")
      MESSAGE(STATUS "Detected Open MPI. Setting bind-to options for thread affinity in MPIEXEC_PREFLAGS.")
    ELSE(NOT MPIEXEC_PREFLAGS)
      IF(NOT "${MPIEXEC_PREFLAGS}" MATCHES "--bind-to")
        MESSAGE(WARNING "Default Open MPI thread affinity policy may bind all the threads to a single core and "
                        "significantly slow down testing. Add proper --bind-to options via MPIEXEC_PREFLAGS. "
                        "If you don't know exactly what affinity to add, "
                        "add '--bind-to;none' to your current MPIEXEC_PREFLAGS to stop this warning.")
      ENDIF()
    ENDIF(NOT MPIEXEC_PREFLAGS)
  ENDIF()

  IF(NOT ${CMAKE_CXX_COMPILER} STREQUAL ${MPI_CXX_COMPILER})
    SET(MPI_WARNING_LIST
        "Building MPI version without using MPI compiler wrappers.\n"
        "This may not build miniqmc correctly. To ensure the correct version, specify the compiler wrappers to cmake.\n"
        "For example: cmake -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpic++\n"
        "To build without MPI, pass '-DQMC_MPI=0' to cmake")
    MESSAGE(WARNING ${MPI_WARNING_LIST})
  ENDIF()

  IF ( NOT TEST_MAX_PROCS )
    SET( TEST_MAX_PROCS 100 )
  ENDIF()

  MESSAGE(STATUS "MPI runner MPIEXEC_EXECUTABLE : ${MPIEXEC_EXECUTABLE}")
  MESSAGE(STATUS "MPIEXEC_NUMPROC_FLAG : ${MPIEXEC_NUMPROC_FLAG}")
  MESSAGE(STATUS "MPIEXEC_PREFLAGS : ${MPIEXEC_PREFLAGS}")
  STRING(REPLACE ";" " " MPIEXEC_PREFLAGS_PRINT "${MPIEXEC_PREFLAGS}")
  MESSAGE(STATUS "Tests run as : ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} NUM_PROCS ${MPIEXEC_PREFLAGS_PRINT} EXECUTABLE")

  SET(HAVE_MPI 1)
  TARGET_LINK_LIBRARIES(qmc_mpi_cxx INTERFACE MPI::MPI_CXX)
  MESSAGE(STATUS "MPI is enabled")
ELSE(QMC_MPI)
  SET(HAVE_MPI 0)
  MESSAGE(STATUS "MPI is disabled")
ENDIF(QMC_MPI)

####################################################################
#First check the required libraries. Abort if these are not found.
#Changed the fall-back versions to use similar conventions to 2.8.0 and higher
####################################################################
# list of libraries to be linked with the main applications using I/O

#-------------------------------------------------------------------
# check lapack/blas
#-------------------------------------------------------------------
# Add Math::BLAS_LAPACK target
ADD_LIBRARY(Math::BLAS_LAPACK INTERFACE IMPORTED)

FOREACH(MKL_LIB_PATH IN ITEMS ${MKL_ROOT} $ENV{MKL_ROOT} $ENV{MKLROOT} $ENV{MKL_HOME})
  SET(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} ${MKL_LIB_PATH}/lib/intel64)
ENDFOREACH()
IF(NOT BLA_VENDOR)
  MESSAGE(STATUS "Trying to find LAPACK from Intel MKL")
  IF(QMC_OMP AND CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    SET(BLA_VENDOR Intel10_64lp)
  ELSE()
    SET(BLA_VENDOR Intel10_64lp_seq)
  ENDIF()
  find_package(LAPACK)
  IF(NOT LAPACK_FOUND)
    MESSAGE(STATUS "Intel MKL library files not found via FindLAPACK.")
    MESSAGE(STATUS "Trying to find alternative LAPACK libraries")
    SET(BLA_VENDOR All)
    find_package(LAPACK REQUIRED)
  ENDIF()
ELSE()
  MESSAGE(STATUS "Trying to find LAPACK library as requested BLA_VENDOR=${BLA_VENDOR}")
  find_package(LAPACK REQUIRED)
ENDIF()

MESSAGE(STATUS "LAPACK linker flags: ${LAPACK_LINKER_FLAGS}")
MESSAGE(STATUS "LAPACK libraries: ${LAPACK_LIBRARIES}")
SET_TARGET_PROPERTIES(Math::BLAS_LAPACK PROPERTIES INTERFACE_LINK_LIBRARIES "${LAPACK_LINKER_FLAGS};${LAPACK_LIBRARIES}")

# Detects MKL header files and other components.
IF(LAPACK_LIBRARIES MATCHES "mkl_core")
  INCLUDE(CMake/FindMKL.cmake)
ENDIF()

# AFQMC requires MKL sparse for good performance (roughly a factor of 2x)
IF (BUILD_AFQMC AND NOT MKL_FOUND)
  MESSAGE(WARNING "AFQMC - MKL not found, using simple sparse matrix routines.  Link with MKL sparse libraries for better performance.")
ENDIF()

SET(QMC_UTIL_LIBS ${QMC_UTIL_LIBS} ${FORTRAN_LIBRARIES})

#find_package(ZLIB)
   
#set(HDF5_USE_STATIC_LIBRARIES off)
#if(QMC_BUILD_STATIC)
#  set(HDF5_USE_STATIC_LIBRARIES on)
#endif()

#find_package(HDF5)
#IF(HDF5_FOUND)
#  SET(HAVE_LIBHDF5 1)
#  INCLUDE_DIRECTORIES(${HDF5_INCLUDE_DIR})
#  IF ( CMAKE_BUILD_TYPE AND HDF5_LIBRARIES_DEBUG )
#    IF ( CMAKE_BUILD_TYPE MATCHES DEBUG )
#      SET( HDF5_LIBRARIES ${HDF5_LIBRARIES_DEBUG} )
#    ELSE()
#      SET( HDF5_LIBRARIES ${HDF5_LIBRARIES_RELEASE} )
#    ENDIF()
#  ENDIF()
#  SET(QMC_UTIL_LIBS ${QMC_UTIL_LIBS} ${HDF5_LIBRARIES})
#  if(ZLIB_LIBRARY)
#    SET(QMC_UTIL_LIBS ${QMC_UTIL_LIBS} ${ZLIB_LIBRARY})
#  endif(ZLIB_LIBRARY)
#  #IF(SZLIB_FOUND)
#  #  SET(QMC_UTIL_LIBS ${QMC_UTIL_LIBS} ${SZLIB_LIBRARIES})
#  #ENDIF(SZLIB_FOUND)
#ELSE(HDF5_FOUND)
#  if(NOT QMC_PHI)
#    MESSAGE(FATAL_ERROR "Require hdf5 1.6.4 or higher. Set HDF5_ROOT")
#  endif()
#ENDIF(HDF5_FOUND)
#

######################################################
# Add optional macros
######################################################

#include miniqmc/src build/src
INCLUDE_DIRECTORIES( ${PROJECT_SOURCE_DIR}/src ${PROJECT_BINARY_DIR}/src)

#  To connect the timer regions with VTune sampling, some Task API calls
#  are present in the code.  To enable them, set USE_VTUNE_TASKS and add the
#  include and lib paths to VTune.
#  An example for what needs to be added to the cmake command:
#
# -DUSE_VTUNE_TASKS=1 \
# -DQMC_INCLUDE=/opt/intel/vtune_amplifier_xe/include \
# -DQMC_EXTRA_LIBS="-L/opt/intel/vtune_amplifier_xe/lib64 -ldl"
#
# Some collection types may require enabling the task API with a knob.
# For the command line:
# amplxe-cl -collect advanced-hostpots -knob enable-user-tasks=true ./bin/miniqmc
# For the GUI, there is an equivalent checkbox in the collection setup.
#

IF (USE_VTUNE_TASKS)
  LINK_LIBRARIES(ittnotify)
ENDIF()

###############################################################
# for specific linker in the case of profiling like hpctoolkit.
###############################################################
if(CMAKE_CXX_LINKER)
  set(CMAKE_CXX_LINK_EXECUTABLE
  "${CMAKE_CXX_LINKER} <FLAGS> <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS>  <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")
  MESSAGE(STATUS "Linker is specified as '${CMAKE_CXX_LINKER}' instead of '${CMAKE_CXX_COMPILER}'")
endif()

###################################################
# Add user extra flags
###################################################
IF(QMC_INCLUDE)
  FOREACH ( tmp ${QMC_INCLUDE} )
    MESSAGE(STATUS "Adding '${tmp}' to include directories")
    INCLUDE_DIRECTORIES( "${tmp}" )
  ENDFOREACH()
ENDIF()
IF(QMC_EXTRA_LIBS)
  string(REPLACE " " ";" QMC_EXTRA_LIBS ${QMC_EXTRA_LIBS})
  SET(QMC_UTIL_LIBS ${QMC_UTIL_LIBS} ${QMC_EXTRA_LIBS})
  MESSAGE("QMC_UTIL_LIBS=${QMC_UTIL_LIBS}")
ENDIF()

# Avoid unused variables warnings (for now) by using for information purposes
IF(QMC_DATA)
MESSAGE(STATUS "QMC_DATA is ${QMC_DATA}")
ENDIF()
IF(ENABLE_GCOV)
MESSAGE(STATUS "GCOV is enabled")
ENDIF()

############################################################
# FLAGS at the project level
############################################################
#this might be redundant but maybe not in all CMake versions.
STRING(TOUPPER "${CMAKE_BUILD_TYPE}" THIS_CONFIG)

SET(PROJECT_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
#pre 3.0 cmake does not have string CONCAT
SET(TMP_PROJECT_CMAKE_CXX_FLAGS "${PROJECT_CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${THIS_CONFIG}}")
MESSAGE("Project CXX_FLAGS: ${TMP_PROJECT_CMAKE_CXX_FLAGS}")

get_directory_property(TMP_PROJECT_INCLUDE_DIRECTORIES INCLUDE_DIRECTORIES)
MESSAGE("Project INCLUDE_DIRECTORIES: ${TMP_PROJECT_INCLUDE_DIRECTORIES}")

FOREACH(target_type IN ITEMS EXE SHARED)
  SET(PROJECT_CMAKE_${target_type}_LINKER_FLAGS ${CMAKE_${target_type}_LINKER_FLAGS})
  SET(TMP_PROJECT_CMAKE_${target_type}_LINKER_FLAGS "${PROJECT_CMAKE_${target_type}_LINKER_FLAGS} ${CMAKE_${target_type}_LINKER_FLAGS_${THIS_CONFIG}}")
  MESSAGE("Project ${target_type}_LINKER_FLAGS: ${TMP_PROJECT_CMAKE_${target_type}_LINKER_FLAGS}")
ENDFOREACH(target_type IN ITEMS EXE SHARED)

###################################################
# Done searching libraries. Start building miniQMC
###################################################

MESSAGE(STATUS "Ready to build miniQMC")

####################################################################
# add definitions
#  -DH5_USE_16_API for HDF5
####################################################################

ADD_DEFINITIONS(-DHAVE_CONFIG_H -DH5_USE_16_API -DADD_)

EXECUTE_PROCESS(COMMAND hostname OUTPUT_VARIABLE QMC_HOSTNAME OUTPUT_STRIP_TRAILING_WHITESPACE)
CONFIGURE_FILE(${miniqmc_SOURCE_DIR}/miniqmc.settings ${miniqmc_BINARY_DIR}/bin/miniqmc.settings)

SUBDIRS(src)

