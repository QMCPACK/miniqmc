######################################################################
# GLOBAL
######################################################################
CMAKE_MINIMUM_REQUIRED(VERSION 3.6.0)
cmake_policy(SET CMP0003 NEW)
#Actually set CXX11 flag for Clang on Apple, even llvm non apple clang
#This is contrary to what the --help-policy says
cmake_policy(SET CMP0025 NEW)

######################################################################
PROJECT(qmcpack CXX)
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
SET( DROP_METHOD "http" )
SET( DROP_SITE "cdash.qmcpack.org" )
SET( DROP_LOCATION "/CDash/submit.php?project=QMCPACK" )
SET( TRIGGER_SITE "" )
SET( DROP_SITE_CDASH TRUE )
# Increase timeout per test over the default of 1500 seconds (25 minutes)
SET( DART_TESTING_TIMEOUT 3600 CACHE STRING "Maximum time for one test")
ENABLE_TESTING()
INCLUDE( CTest )

######################################################################
# Build level
######################################################################
SET(QMC_BUILD_LEVEL 5 CACHE INTEGER
  "QMC Build Level: 1=bare, 2=developer, 3=experimental, 4=minimal, 5=miniapps")
IF ( NOT CMAKE_BUILD_TYPE AND NOT CMAKE_TOOLCHAIN_FILE)
  SET( CMAKE_BUILD_TYPE Release )
ENDIF()
STRING( TOUPPER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE )
MESSAGE( STATUS "CMAKE_BUILD_TYPE is ${CMAKE_BUILD_TYPE}")

######################################################################
#define the paths for library and executable for out-of-soruce compilation
######################################################################
SET (LIBRARY_OUTPUT_PATH ${qmcpack_BINARY_DIR}/lib CACHE PATH "Single output directory for building all libraries.")
SET (EXECUTABLE_OUTPUT_PATH ${qmcpack_BINARY_DIR}/bin CACHE PATH "Single output directory for building all executables.")

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
SET(OHMMS_DIM 3 CACHE INTEGER "Select physical dimension")
SET(OHMMS_INDEXTYPE int)
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
SET(QMC_COMPLEX 0 CACHE INTEGER "Build for complex binary")

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
SET(BUILD_FCIQMC 0 CACHE BOOL "Build with FCIQMC")
SET(QMC_BUILD_STATIC 0 CACHE BOOL "Link to static libraries")
SET(ENABLE_TIMERS 1 CACHE BOOL "Enable internal timers")

######################################################################
# FIXED PARAMETERS for test and legacy reasons
# DO NOT TOUCH THESE
# QMC_ORTHO  = 0 for general unit cell, 1 for orthogonal cells
# QMC_FASTWALKER  = true, when operation is faster for each walker
######################################################################
SET(QMC_FASTWALKER 1)
SET(OHMMS_ORTHO 0)

######################################################################
#list of options to determine libraries. First, try to use libraries available
#FindXYZ.cmake will set the values if successful
######################################################################
#SET(ENABLE_PHDF5 0 CACHE BOOL "Enable phdf5 for output")
#SET(HAVE_LIBHDF5 0)

######################################################
# Directory where customize cmake files reside
######################################################
SET (PROJECT_CMAKE ${qmcpack_SOURCE_DIR}/CMake)

SET(QMCPACK_UNIT_TEST_DIR ${qmcpack_BINARY_DIR}/tests/bin)

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
  SET(HAVE_MPI 0)
  SET(INTEL_COMPILER 0)
  SET(HAVE_MKL 0)
  SET(HAVE_MKL_VML 0)
  SET(HAVE_ACML 0)

  ######################################################
  # Try to identify CPU identity
  ######################################################
  SET(CPU_IDENTITY "generic")
  #INCLUDE(${PROJECT_CMAKE}/CheckProcessorID.cmake)

  #------------------------
  # On Cray's machine
  #------------------------
  IF($ENV{CRAYPE_VERSION} MATCHES ".")
    MESSAGE(STATUS "Running on a Cray machine.")
    SET(CMAKE_SKIP_RPATH TRUE)

    # Flags for ctest
    SET(USE_EXT_MPI_FOR_SERIAL_TESTS 1)
    FIND_PROGRAM(MPIEXEC aprun)
    IF(NOT MPIEXEC)
    # For slurm use srun
      FIND_PROGRAM(MPIEXEC srun)
    ENDIF(NOT MPIEXEC)
    SET(MPIEXEC_NUMPROC_FLAG "-n")

    IF(QMC_BUILD_STATIC)
      MESSAGE(STATUS "Building CPU code. Linking statically.")
      SET(BUILD_SHARED_LIBS off)
      SET(CMAKE_FIND_LIBRARY_SUFFIXES .a)
      FOREACH(type SHARED_LIBRARY SHARED_MODULE EXE)
        SET(CMAKE_${type}_LINK_STATIC_CXX_FLAGS "-Wl,-Bstatic")
        SET(CMAKE_${type}_LINK_DYNAMIC_CXX_FLAGS "-static")
      ENDFOREACH(type)
    ELSE(QMC_BUILD_STATIC)
      MESSAGE(STATUS "Building CPU code. Linking dynamically.")
    ENDIF(QMC_BUILD_STATIC)
  ENDIF($ENV{CRAYPE_VERSION} MATCHES ".")

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
  ELSE()
    MESSAGE(WARNING "No default file for compiler (${COMPILER})")
  ENDIF()

  # requires C++11 standard
  SET(CMAKE_CXX_STANDARD 11)
  SET(CMAKE_CXX_STANDARD_REQUIRED ON)
  SET(CMAKE_CXX_EXTENSIONS OFF)

  #-------------------------------------------------------------------
  #  check MPI installation
  #-------------------------------------------------------------------
  IF(QMC_MPI)
    ## mpi compilers
    GET_FILENAME_COMPONENT(BASE_CXX_COMPILER_NAME ${CMAKE_CXX_COMPILER} NAME)
    if(${BASE_CXX_COMPILER_NAME} MATCHES "^mp")
      SET(HAVE_MPI 1)
      SET(MPI_FOUND TRUE)
    elseif($ENV{CRAYPE_VERSION} MATCHES ".")
      SET(HAVE_MPI 1)
      SET(MPI_FOUND TRUE)
    else()
      include(FindMPI)
      IF(MPI_FOUND)
        MESSAGE(STATUS "Found mpi.h and libraries. Check for working mpi compiler ")
        INCLUDE_DIRECTORIES(${MPI_CXX_INCLUDE_PATH})
        FILE(WRITE ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/testmpi.cxx
          "#include <mpi.h>"
          "int main(int argc, char** argv){MPI_Init(&argc,&argv); MPI_Finalize();return 0;}\n")
        #TRY_COMPILE(MPI_WORKS ${CMAKE_BINARY_DIR}
        #${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/testmpi.cxx
        #OUTPUT_VARIABLE OUTPUT)
        set(MPI_WORKS 1)
        SET(MPI_WARNING_LIST
            "Building MPI version without using MPI compiler wrappers.\n"
            "This may not build qmcpack correctly. To ensure the correct version, specify the compiler wrappers to cmake.\n"
            "For example: cmake -DCMAKE_CXX_COMPILER=mpic++\n"
            "To build without MPI, pass '-DQMC_MPI=0' to cmake")
        MESSAGE(WARNING ${MPI_WARNING_LIST})

        IF(MPI_WORKS)
          MESSAGE(STATUS "MPI is enabled")
          SET(HAVE_MPI 1)
          LINK_LIBRARIES(${MPI_CXX_LIBRARIES})
        ELSE(MPI_WORKS)
          MESSAGE(STATUS "MPI is disabled")
          SET(HAVE_MPI 0)
          SET(MPI_LIBRARY "")
        ENDIF(MPI_WORKS)
      ELSE(MPI_FOUND)
        MESSAGE(STATUS "MPI not found")
        SET(MPI_LIBRARY "")
      ENDIF(MPI_FOUND)
    ENDIF()
  ENDIF(QMC_MPI)

  #-------------------------------------------------------------------
  # check lapack/blas
  #-------------------------------------------------------------------
  # Quick search for MKL (this will need to be improved).
  # This reproduces the old intel -mkl behavior or
  # if ENABLE_MKL is set to 1 supports other compilers
  IF (INTEL_COMPILER)
    SET(ENABLE_MKL TRUE CACHE BOOL "ENABLE MKL for Intel compilers")
  ELSE (INTEL_COMPILER)
    SET(ENABLE_MKL FALSE CACHE BOOL "ENABLE MKL for non-intel compilers (Experimental)")
  ENDIF (INTEL_COMPILER)
  IF (ENABLE_MKL)
   INCLUDE(CMake/FindMKL.cmake)
  ENDIF(ENABLE_MKL)
  IF ( HAVE_MKL )
    # We found MKL, add the link library
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MKL_FLAGS}")
    SET(LAPACK_FOUND 1)
    SET(LAPACK_LIBRARIES "${MKL_LIBRARIES}")
    SET(LAPACK_LINKER_FLAGS "${MKL_LINKER_FLAGS}")
  ELSEIF ( $ENV{CRAYPE_VERSION} MATCHES "." )
    # Cray specific environment variable indicates we are on a cray. Assume libsci will be linked
    # preferable to attempt a compile/link 
    SET(LAPACK_FOUND 1)
    MESSAGE(STATUS "Assuming Cray libsci or other BLAS/LAPACK will be linked by compiler wrappers")
  ELSE()	 
    # Search for Blas/Lapack
    find_package(LAPACK)
    # When cmake finds and sets up the MKL libraries with a non-Intel compiler, set the flag
    # This should be tidied up when FindMKL.cmake supports non-Intel compilers.
    IF("${LAPACK_LIBRARIES}" MATCHES "mkl")
      MESSAGE(STATUS "MKL found via LAPACK/BLAS")
      SET( MKL_FOUND 1 )
      SET( HAVE_MKL 1 )
    ENDIF()
  ENDIF()
  IF(LAPACK_FOUND)
    MESSAGE(STATUS "LAPACK libraries: ${LAPACK_LIBRARIES}")
    MESSAGE(STATUS "LAPACK linker flags: ${LAPACK_LINKER_FLAGS}")
    SET(LAPACK_LIBRARY ${LAPACK_LIBRARIES} ${LAPACK_LINKER_FLAGS})
  ELSE()
    MESSAGE(FATAL_ERROR "Could not find required libraries LAPACK &/or BLAS") 
  ENDIF()

  INCLUDE(CMake/FindVectorMath.cmake)

ENDIF(CMAKE_TOOLCHAIN_FILE)

IF (ENABLE_GCOV)
  IF (NOT GCOV_SUPPORTED)
    MESSAGE(FATAL_ERROR "ENABLE_GCOV was specified but compiler does not support GCC coverage flag")
  ENDIF()
ENDIF(ENABLE_GCOV)

# AFQMC requires MKL sparse for good performance (roughly a factor of 2x)
IF (BUILD_AFQMC AND NOT MKL_FOUND)
  MESSAGE(WARNING "AFQMC - MKL not found, using simple sparse matrix routines.  Link with MKL sparse libraries for better performance.")
ENDIF()

# AFQMC requires MPI
If (BUILD_AFQMC AND NOT QMC_MPI)
  MESSAGE(FATAL_ERROR "AFQMC requires building with MPI (QMC_MPI=1). Set BUILD_AFQMC=0 or configure MPI.")
ENDIF()

IF (BUILD_AFQMC AND NOT APPLE)
  LINK_LIBRARIES("rt")
ENDIF()
  
# setup ctest variables
IF ( HAVE_MPI )
  SET( USE_MPI 1 )
  IF ( NOT MPIEXEC )
    SET( MPIEXEC "mpirun" )
  ENDIF()

  IF ( NOT DEFINED MPIEXEC_NUMPROC_FLAG )
    SET( MPIEXEC_NUMPROC_FLAG "-np" )
  ENDIF()

  IF ( NOT TEST_MAX_PROCS )
    SET( TEST_MAX_PROCS 100 )
  ENDIF()

  MESSAGE(STATUS "MPI runner is ${MPIEXEC}")
ENDIF(HAVE_MPI)

MESSAGE(STATUS "CXX compiler flags: ${CMAKE_CXX_FLAGS}")

INCLUDE(${CMAKE_ROOT}/Modules/FindThreads.cmake)

####################################################################
#First check the required libraries. Abort if these are not found.
#Changed the fall-back versions to use similar conventions to 2.8.0 and higher
####################################################################
# list of libraries to be linked with the main applications using I/O

SET(QMC_UTIL_LIBS ${LAPACK_LIBRARY} ${BLAS_LIBRARY})
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

#include qmcpack/src build/src
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

###################################################
# Done searching libraries. Start building qmcpack
###################################################

MESSAGE(STATUS "Ready to build qmcpack")

####################################################################
# add definitions
#  -DH5_USE_16_API for HDF5
#  -DMPICH_SKIP_MPICXX : disable C++ binding of MPICH
####################################################################

ADD_DEFINITIONS(-DHAVE_CONFIG_H -DH5_USE_16_API -DUSE_REAL_STRUCT_FACTOR -DINLINE_ALL=inline -DADD_)

if(HAVE_MPI)
  ADD_DEFINITIONS(-DMPICH_SKIP_MPICXX)
endif()

EXECUTE_PROCESS(COMMAND hostname OUTPUT_VARIABLE QMC_HOSTNAME OUTPUT_STRIP_TRAILING_WHITESPACE)
CONFIGURE_FILE(${qmcpack_SOURCE_DIR}/qmcpack.settings ${qmcpack_BINARY_DIR}/bin/qmcpack.settings)

SUBDIRS(src)

