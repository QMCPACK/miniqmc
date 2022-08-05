######################################################################
# GLOBAL
######################################################################
cmake_minimum_required(VERSION 3.17.0)
cmake_policy(SET CMP0003 NEW)
#Actually set CXX11 flag for Clang on Apple, even llvm non apple clang
#This is contrary to what the --help-policy says
cmake_policy(SET CMP0025 NEW)
include(CMakeDependentOption)

######################################################################
project(miniqmc CXX)
######################################################################

######################################################################
# Version
######################################################################
set(QMCPACK_VERSION_MAJOR 3)
set(QMCPACK_VERSION_MINOR 1)
set(QMCPACK_VERSION_PATCH 0)
set(QMCPACK_VERSION "${QMCPACK_VERSION_MAJOR}.${QMCPACK_VERSION_MINOR}.${QMCPACK_VERSION_PATCH}")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_TOOLCHAIN_FILE)
  set(CMAKE_BUILD_TYPE Release)
endif()
string(TOUPPER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE)
message(STATUS "CMAKE_BUILD_TYPE is ${CMAKE_BUILD_TYPE}")

######################################################################
# Define the paths for library and executable for out-of-soruce compilation
# Directory where customize cmake files reside
######################################################################
set(PROJECT_CMAKE ${miniqmc_SOURCE_DIR}/CMake)

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
set(OHMMS_DIM
    3
    CACHE STRING "Select physical dimension")
set(OHMMS_INDEXTYPE int)
set(OHMMS_ORTHO 0)
message(STATUS "defining the float point precision")
set(OHMMS_PRECISION_FULL double)
set(QMC_MIXED_PRECISION
    0
    CACHE BOOL "Enable/disable mixed precision")
if(QMC_MIXED_PRECISION)
  set(OHMMS_PRECISION float)
  set(MIXED_PRECISION 1)
else(QMC_MIXED_PRECISION)
  set(OHMMS_PRECISION double)
endif(QMC_MIXED_PRECISION)
message("   Base precision = ${OHMMS_PRECISION}")
message("   Full precision = ${OHMMS_PRECISION_FULL}")

# Code coverage
set(GCOV_SUPPORTED FALSE)
set(ENABLE_GCOV
    FALSE
    CACHE BOOL "Enable code coverage")

######################################################################
# enable MPI and OPNEMP  if possible
######################################################################
set(QMC_MPI
    0
    CACHE BOOL "Enable/disable MPI")
set(QMC_OMP
    1
    CACHE BOOL "Enable/disable OpenMP")
set(QMC_COMPLEX
    0
    CACHE STRING "Build for complex binary")

######################################################################
# Standard test
######################################################################
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake" ${CMAKE_MODULE_PATH})

######################################################################
# add macros to compile einspline if necessary
######################################################################
include(${CMAKE_ROOT}/Modules/CheckFunctionExists.cmake)

check_function_exists(sincos HAVE_SINCOS)
check_function_exists(posix_memalign HAVE_POSIX_MEMALIGN)

######################################################################
# Build options
# QMC_BUILD_STATIC build static/dynamic  libraries
# BUILD_QMCTOOLS   build utility programs
# BUILD_SANDBOX    build test programs
######################################################################
set(BUILD_UNIT_TESTS
    1
    CACHE BOOL "Build unit tests")
set(BUILD_LMYENGINE_INTERFACE
    1
    CACHE BOOL "Build LMY engine")
if(MIXED_PRECISION AND BUILD_LMYENGINE_INTERFACE)
  message(STATUS "LMY engine is not compatiable with CPU mixed precision build! Disabling LMY engine")
  set(BUILD_LMYENGINE_INTERFACE 0)
endif()
set(BUILD_AFQMC
    0
    CACHE BOOL "Build with AFQMC")
# AFQMC requires MPI
if(BUILD_AFQMC AND NOT QMC_MPI)
  message(FATAL_ERROR "AFQMC requires building with MPI (QMC_MPI=1). Set BUILD_AFQMC=0 or configure MPI.")
endif()

set(BUILD_FCIQMC
    0
    CACHE BOOL "Build with FCIQMC")
set(QMC_BUILD_STATIC
    0
    CACHE BOOL "Link to static libraries")
set(ENABLE_TIMERS
    1
    CACHE BOOL "Enable internal timers")
option(ENABLE_OFFLOAD "Enable OpenMP offload" OFF)
option(QMC_ENABLE_ROCM "Enable GPU support through ROCM" OFF)
if(QMC_ENABLE_ROCM)
  set(QMC_ENABLE_CUDA ON)
  set(QMC_CUDA2HIP ON)
endif()
option(QMC_ENABLE_CUDA "Enable CUDA acceleration" OFF)
option(QMC_ENABLE_SYCL "Enable SYCL offload" OFF)
# Use CMake object library targets to workaround clang linker not being able to handle fat
# binary archives which contain both host and device codes, for example OpenMP offload regions.
# CMake does not propagate indirect object files by design.
# So the dependency must be explicitly specified with USE_OBJECT_TARGET.
cmake_dependent_option(USE_OBJECT_TARGET "Use CMake object library target" OFF "ENABLE_OFFLOAD" OFF)

######################################################################
# Sanitizer options
######################################################################

# Add optional sanitizers ASAN, UBSAN, MSAN
set(VALID_SANITIZERS "none" "asan" "ubsan" "tsan" "msan")
set(ENABLE_SANITIZER
    "none"
    CACHE STRING "none,asan,ubsan,tsan,msan")
set_property(CACHE ENABLE_SANITIZER PROPERTY STRINGS ${VALID_SANITIZERS})

# Perform sanitizer option check, only works in debug mode
if(NOT ENABLE_SANITIZER IN_LIST VALID_SANITIZERS)
  message(FATAL_ERROR "Invalid -DENABLE_SANITIZER=${ENABLE_SANITIZER}, value must be one of ${VALID_SANITIZERS}")
else()
  message(STATUS "Enable sanitizer ENABLE_SANITIZER=${ENABLE_SANITIZER}")
endif()

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
find_package(Git)

if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
  set(IS_GIT_PROJECT 1)
endif()

######################################################################
# Set the compiler specific options/flags
######################################################################
if(CMAKE_TOOLCHAIN_FILE)
  message(STATUS "Using ${CMAKE_TOOLCHAIN_FILE} toolchain ")
else(CMAKE_TOOLCHAIN_FILE)
  message(STATUS "Trying to figure out compiler options ....")
  set(ENABLE_OPENMP 0)

  #------------------------
  # On Cray's machine
  #------------------------
  get_filename_component(BASE_CXX_COMPILER_NAME ${CMAKE_CXX_COMPILER} NAME)
  if($ENV{CRAYPE_VERSION} MATCHES "." AND NOT BASE_CXX_COMPILER_NAME STREQUAL "CC")
    message(WARNING "Cray Programming Environment has been detected but C++ compiler wrapper CC is not used! "
                    "Please use cc/CC as the C/C++ compiler to ensure correct cross-compiling for the compute nodes "
                    "unless you understand cross-compiling and intend not to use Cray compiler wrappers.")
  endif()
  if(CMAKE_SYSTEM_NAME STREQUAL "CrayLinuxEnvironment")
    message(STATUS "Running on a Cray machine.")

    if(NOT $ENV{CRAYPE_LINK_TYPE} STREQUAL "dynamic")
      message(WARNING "Cray Programming Environment uses static linking by default. "
                      "We prefer dynamic linking which makes library searching easier. "
                      "Setting environment variable CRAYPE_LINK_TYPE to \"dynamic\" enables dynamic linking.\n")
    endif()

    if(NOT DEFINED MPIEXEC_EXECUTABLE)
      find_program(MPIEXEC_EXECUTABLE aprun)
      if(NOT MPIEXEC_EXECUTABLE)
        # For slurm use srun
        find_program(MPIEXEC_EXECUTABLE srun)
      endif(NOT MPIEXEC_EXECUTABLE)
    endif()
  else()
    if($ENV{CRAYPE_VERSION} MATCHES "." AND BASE_CXX_COMPILER_NAME STREQUAL "CC")
      message(
        FATAL_ERROR
          "Cray compiler wrapper detected. Empty the build folder and rerun cmake with -DCMAKE_SYSTEM_NAME=CrayLinuxEnvironment added."
      )
    endif()
  endif()

  include(inspectCompiler)

endif(CMAKE_TOOLCHAIN_FILE)

# requires C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(ENABLE_GCOV)
  if(NOT GCOV_SUPPORTED)
    message(FATAL_ERROR "ENABLE_GCOV was specified but compiler does not support GCC coverage flag")
  endif()
endif(ENABLE_GCOV)

if(BUILD_AFQMC AND NOT APPLE)
  link_libraries("rt")
endif()

#-------------------------------------------------------------------------------
# Check SIMD alignment for CPU only
#-------------------------------------------------------------------------------
# This is intentionally placed before adding OpenMP offload compile options
# to avoid contamination from device compilation pass.
# When '-march=skylake-avx512 -fopenmp-targets=nvptx64 -march=sm_70' is added
# for Clang, the source code is parsed twice for both host and offload targets.
# A trap for macro __AVX512F__ always fails because the offload pass doesn't
# carry '-march=skylake-avx512' but only takes '-march=sm_70'.
#-------------------------------------------------------------------------------
include(CMake/CheckSIMDAlignment.cmake)
set(QMC_SIMD_ALIGNMENT
    ${default_alignment}
    CACHE STRING "Cache/SIMD alignment in bytes")
math(EXPR alignment_remainder "${QMC_SIMD_ALIGNMENT} % ${default_alignment}")
if(alignment_remainder)
  message(
    FATAL_ERROR
      "QMC_SIMD_ALIGNMENT must be a multiple of ${default_alignment}. Bad cached value is ${QMC_SIMD_ALIGNMENT}")
endif()
message(STATUS "QMC_SIMD_ALIGNMENT is set to ${QMC_SIMD_ALIGNMENT}")

#---------------------------------------------------------
# Determine if OpenMP taskloop works with the CXX compiler
#---------------------------------------------------------
include(CMake/TestOpenMPtaskloop.cmake)
option(ENABLE_OMP_TASKLOOP "Enable OpenMP taskloop" ${OMP_TASKLOOP_OKAY})
message(STATUS "ENABLE_OMP_TASKLOOP is set to ${ENABLE_OMP_TASKLOOP}")

#---------------------------------------------------------
# Set up OpenMP offload compile options
#---------------------------------------------------------
set(QMC_OFFLOAD_MEM_ASSOCIATED_DEFAULT OFF)
if(ENABLE_OFFLOAD AND DEFINED OPENMP_OFFLOAD_COMPILE_OPTIONS)
  message(STATUS "OpenMP offload CXX flags: ${OPENMP_OFFLOAD_COMPILE_OPTIONS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OPENMP_OFFLOAD_COMPILE_OPTIONS}")
  if(${COMPILER} MATCHES "Clang"
     AND OPENMP_OFFLOAD_COMPILE_OPTIONS MATCHES "amdgcn"
     AND QMC_CUDA2HIP)
    # As of 11/2021, QMC_OFFLOAD_MEM_ASSOCIATED=ON is needed for AMD and mainline LLVM compilers
    # when using OpenMP offload to AMD GPU together with HIP.
    set(QMC_OFFLOAD_MEM_ASSOCIATED_DEFAULT ON)
  endif()
endif()
# Some OpenMP offload runtime libraries have composibility issue with a vendor native runtime.
# A workaround is making the vendor native runtime responsible for memory allocations and OpenMP associate/disassocate them.
cmake_dependent_option(QMC_OFFLOAD_MEM_ASSOCIATED "Manage OpenMP memory allocations via the vendor runtime"
                       ${QMC_OFFLOAD_MEM_ASSOCIATED_DEFAULT} "ENABLE_OFFLOAD;QMC_ENABLE_CUDA" OFF)

#-------------------------------------------------------------------
# Check MPI installation. MPI is treated as a part of compiler
#-------------------------------------------------------------------
add_library(qmc_mpi_cxx INTERFACE)
if(QMC_MPI)
  # for backward compatibility with MPIEXEC
  if(DEFINED MPIEXEC AND NOT DEFINED MPIEXEC_EXECUTABLE)
    message(
      WARNING
        "Setting MPIEXEC_EXECUTABLE based on MPIEXEC. MPIEXEC has been deprecated! Use MPIEXEC_EXECUTABLE instead!")
    set(MPIEXEC_EXECUTABLE ${MPIEXEC})
  endif(DEFINED MPIEXEC AND NOT DEFINED MPIEXEC_EXECUTABLE)
  set(MPI_CXX_SKIP_MPICXX
      TRUE
      CACHE BOOL "Must be TRUE for QMCPACK. Cached for working around CMake FindMPI issue.")
  if(NOT CMAKE_CROSSCOMPILING)
    set(MPI_DETERMINE_LIBRARY_VERSION TRUE)
  endif()
  find_package(MPI COMPONENTS CXX)

  if(NOT MPI_FOUND)
    message(
      FATAL_ERROR
        "MPI support not found! Provide MPI compiler wrappers or build without MPI by passing '-DQMC_MPI=0' to cmake.")
  endif(NOT MPI_FOUND)

  if(${MPI_CXX_LIBRARY_VERSION_STRING} MATCHES "MVAPICH2")
    string(REGEX REPLACE "\n" ";" ONE_LINE "${MPI_CXX_LIBRARY_VERSION_STRING}")
    string(REGEX REPLACE " +|\t" ";" ONE_LINE "${ONE_LINE}")
    list(GET ONE_LINE 3 MVAPICH2_VERSION)
    message(STATUS "MVAPICH2 version ${MVAPICH2_VERSION}")
    if(${MVAPICH2_VERSION} VERSION_LESS "2.4" AND NOT ${MPI_CXX_LIBRARY_VERSION_STRING} MATCHES
                                                  "disable-registration-cache")
      message(
        FATAL_ERROR
          "MVAPICH2 (version < 2.4) with registration cache enabled breaks QMCPACK. "
          "Use a different MPI library or a MVAPICH2 >=2.4 "
          "or build MVAPICH2 with --disable-registration-cache configure option.")
    endif()
  endif()

  if(${MPI_CXX_LIBRARY_VERSION_STRING} MATCHES "Open MPI")
    if(NOT MPIEXEC_PREFLAGS)
      set(MPIEXEC_PREFLAGS "--bind-to;none")
      message(STATUS "Detected Open MPI. Setting bind-to options for thread affinity in MPIEXEC_PREFLAGS.")
    else(NOT MPIEXEC_PREFLAGS)
      if(NOT "${MPIEXEC_PREFLAGS}" MATCHES "--bind-to")
        message(
          WARNING "Default Open MPI thread affinity policy may bind all the threads to a single core and "
                  "significantly slow down testing. Add proper --bind-to options via MPIEXEC_PREFLAGS. "
                  "If you don't know exactly what affinity to add, "
                  "add '--bind-to;none' to your current MPIEXEC_PREFLAGS to stop this warning.")
      endif()
    endif(NOT MPIEXEC_PREFLAGS)
  endif()

  if(NOT ${CMAKE_CXX_COMPILER} STREQUAL ${MPI_CXX_COMPILER})
    set(MPI_WARNING_LIST
        "Building MPI version without using MPI compiler wrappers.\n"
        "This may not build qmcpack correctly. To ensure the correct version, specify the compiler wrappers to cmake.\n"
        "For example: cmake -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpic++\n"
        "To build without MPI, pass '-DQMC_MPI=0' to cmake")
    message(WARNING ${MPI_WARNING_LIST})
  endif()

  message(STATUS "MPI runner MPIEXEC_EXECUTABLE : ${MPIEXEC_EXECUTABLE}")
  message(STATUS "MPIEXEC_NUMPROC_FLAG : ${MPIEXEC_NUMPROC_FLAG}")
  message(STATUS "MPIEXEC_PREFLAGS : ${MPIEXEC_PREFLAGS}")
  string(REPLACE ";" " " MPIEXEC_PREFLAGS_PRINT "${MPIEXEC_PREFLAGS}")
  message(
    STATUS "Tests run as : ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} NUM_PROCS ${MPIEXEC_PREFLAGS_PRINT} EXECUTABLE"
  )

  set(HAVE_MPI 1)
  target_link_libraries(qmc_mpi_cxx INTERFACE MPI::MPI_CXX)
  message(STATUS "MPI is enabled")
else(QMC_MPI)
  set(HAVE_MPI 0)
  message(STATUS "MPI is disabled")
endif(QMC_MPI)

#-------------------------------------------------------------------
# final test for compile options before searching for libraries
#-------------------------------------------------------------------
include(CheckCXXSourceCompiles)
check_cxx_source_compiles("int main(){}" TEST_CXX_COMPILE_MAIN)
if (NOT TEST_CXX_COMPILE_MAIN)
  unset(TEST_CXX_COMPILE_MAIN CACHE)
  message(FATAL_ERROR "Failed in compiling a main() function likely due to incorrect compile options. "
                      "Check error in \"${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/CMakeError.log\".")
endif()

####################################################################
#First check the required libraries. Abort if these are not found.
#Changed the fall-back versions to use similar conventions to 2.8.0 and higher
####################################################################
# list of libraries to be linked with the main applications using I/O

if(QMC_ENABLE_CUDA AND NOT QMC_CUDA2HIP)
  find_package(CUDAToolkit)
endif()

#-------------------------------------------------------------------
#  set up ROCM compiler options and libraries
#-------------------------------------------------------------------
if(QMC_ENABLE_ROCM)
  # use FindHIP.cmake to confirm ROCM_ROOT but FindHIP.cmake module is not used
  if(ROCM_ROOT)
    message(STATUS "Using ROCM_ROOT: ${ROCM_ROOT}")
    list(PREPEND CMAKE_PREFIX_PATH ${ROCM_ROOT})
  endif()
  find_package(hip CONFIG REQUIRED)
  find_package(hipblas CONFIG REQUIRED)
  find_package(rocsolver CONFIG REQUIRED)
endif(QMC_ENABLE_ROCM)

#-------------------------------------------------------------------
#  set up SYCL compiler options and libraries
#-------------------------------------------------------------------
if(QMC_ENABLE_SYCL)
  # require 3.20 to recognize IntelLLVM compiler ID and check accurate version numbers.
  if(CMAKE_VERSION VERSION_LESS 3.20.0)
    message(FATAL_ERROR "QMC_ENABLE_SYCL require CMake 3.20.0 or later")
  endif()
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
    message(FATAL_ERROR "QMCPACK only supports SYCL with LLVM-based Intel compiler (icpx).")
  endif()
  add_library(SYCL::host INTERFACE IMPORTED)
  add_library(SYCL::device INTERFACE IMPORTED)
  find_package(IntelDPCPP REQUIRED CONFIGS IntelDPCPPConfig-modified.cmake PATHS ${PROJECT_CMAKE})
  target_link_libraries(SYCL::host INTERFACE OneAPI::DPCPP-host)
  target_link_libraries(SYCL::device INTERFACE OneAPI::DPCPP-device)
endif(QMC_ENABLE_SYCL)

#-------------------------------------------------------------------
# Check LAPACK/BLAS
#-------------------------------------------------------------------
# Add Math::BLAS_LAPACK target
add_library(Math::BLAS_LAPACK INTERFACE IMPORTED)

foreach(MKL_LIB_PATH IN ITEMS ${MKL_ROOT} $ENV{MKL_ROOT} $ENV{MKLROOT} $ENV{MKL_HOME})
  set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} ${MKL_LIB_PATH}/lib/intel64)
endforeach()
if(NOT BLA_VENDOR)
  message(STATUS "Trying to find LAPACK from Intel MKL")
  if(QMC_OMP AND COMPILER STREQUAL "Intel")
    set(BLA_VENDOR Intel10_64lp)
  else()
    set(BLA_VENDOR Intel10_64lp_seq)
  endif()
  find_package(LAPACK)
  if(NOT LAPACK_FOUND)
    message(STATUS "Intel MKL library files not found via FindLAPACK.")
    message(STATUS "Trying to find alternative LAPACK libraries")
    set(BLA_VENDOR All)
    find_package(LAPACK REQUIRED)
  endif()
else()
  message(STATUS "Trying to find LAPACK library as requested BLA_VENDOR=${BLA_VENDOR}")
  find_package(LAPACK REQUIRED)
endif()

message(STATUS "LAPACK linker flags: ${LAPACK_LINKER_FLAGS}")
message(STATUS "LAPACK libraries: ${LAPACK_LIBRARIES}")
target_link_libraries(Math::BLAS_LAPACK INTERFACE "${LAPACK_LINKER_FLAGS};${LAPACK_LIBRARIES}")

#create scalar_vector_functions target
add_library(Math::scalar_vector_functions INTERFACE IMPORTED)
#create fftw3 target
add_library(Math::FFTW3 INTERFACE IMPORTED)

# Detects MKL header files and other components.
if(LAPACK_LIBRARIES MATCHES "mkl_core")
  include(FindMKL)
endif()

if(MKL_FOUND)
  target_link_libraries(Math::BLAS_LAPACK INTERFACE MKL::BLAS_LAPACK)
endif()

set(QMC_UTIL_LIBS ${QMC_UTIL_LIBS} ${FORTRAN_LIBRARIES})

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
include_directories(${PROJECT_SOURCE_DIR}/src ${PROJECT_BINARY_DIR}/src)

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

if(USE_VTUNE_TASKS)
  link_libraries(ittnotify)
endif()

###############################################################
# for specific linker in the case of profiling like hpctoolkit.
###############################################################
if(CMAKE_CXX_LINKER)
  set(CMAKE_CXX_LINK_EXECUTABLE
      "${CMAKE_CXX_LINKER} <FLAGS> <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS>  <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")
  message(STATUS "Linker is specified as '${CMAKE_CXX_LINKER}' instead of '${CMAKE_CXX_COMPILER}'")
endif()

###################################################
# Add user extra flags
###################################################
if(QMC_INCLUDE)
  foreach(tmp ${QMC_INCLUDE})
    message(STATUS "Adding '${tmp}' to include directories")
    include_directories("${tmp}")
  endforeach()
endif()
if(QMC_EXTRA_LIBS)
  string(REPLACE " " ";" QMC_EXTRA_LIBS ${QMC_EXTRA_LIBS})
  set(QMC_UTIL_LIBS ${QMC_UTIL_LIBS} ${QMC_EXTRA_LIBS})
  message("QMC_UTIL_LIBS=${QMC_UTIL_LIBS}")
endif()

# Avoid unused variables warnings (for now) by using for information purposes
if(QMC_DATA)
  message(STATUS "QMC_DATA is ${QMC_DATA}")
endif()
if(ENABLE_GCOV)
  message(STATUS "GCOV is enabled")
endif()

# SETUP SANITIZERS FLAGS
if(NOT "${ENABLE_SANITIZER}" STREQUAL "none")
  if(NOT ${COMPILER} MATCHES "GNU" AND NOT ${COMPILER} MATCHES "Clang")
    message(FATAL_ERROR "-DENABLE_SANITIZER=${ENABLE_SANITIZER} only works with GNU or Clang compilers")
  endif()

  if("${ENABLE_SANITIZER}" STREQUAL "asan")
    set(CMAKE_CXX_FLAGS_SAN
        "-fsanitize=address -fno-optimize-sibling-calls -fsanitize-address-use-after-scope -fno-omit-frame-pointer"
        CACHE STRING "AddressSanitizer C++ compiler builds." FORCE)
  elseif("${ENABLE_SANITIZER}" STREQUAL "ubsan")
    set(CMAKE_CXX_FLAGS_SAN
        "-fsanitize=undefined"
        CACHE STRING "UndefinedBehaviorSanitizer C++ compiler builds." FORCE)
  elseif("${ENABLE_SANITIZER}" STREQUAL "msan")
    set(CMAKE_CXX_FLAGS_SAN
        "-fsanitize=memory"
        CACHE STRING "MemorySanitizer C++ compiler builds." FORCE)
  elseif("${ENABLE_SANITIZER}" STREQUAL "tsan")
    set(CMAKE_CXX_FLAGS_SAN
        "-fsanitize=thread"
        CACHE STRING "ThreadSanitizer C++ compiler builds." FORCE)
  endif()

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_SAN}")

endif()

############################################################
# FLAGS at the project level
############################################################
#this might be redundant but maybe not in all CMake versions.
string(TOUPPER "${CMAKE_BUILD_TYPE}" THIS_CONFIG)

set(PROJECT_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
#pre 3.0 cmake does not have string CONCAT
set(TMP_PROJECT_CMAKE_CXX_FLAGS "${PROJECT_CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${THIS_CONFIG}}")
message("Project CXX_FLAGS: ${TMP_PROJECT_CMAKE_CXX_FLAGS}")

get_directory_property(TMP_PROJECT_INCLUDE_DIRECTORIES INCLUDE_DIRECTORIES)
message("Project INCLUDE_DIRECTORIES: ${TMP_PROJECT_INCLUDE_DIRECTORIES}")

foreach(target_type IN ITEMS EXE SHARED)
  set(PROJECT_CMAKE_${target_type}_LINKER_FLAGS ${CMAKE_${target_type}_LINKER_FLAGS})
  set(TMP_PROJECT_CMAKE_${target_type}_LINKER_FLAGS
      "${PROJECT_CMAKE_${target_type}_LINKER_FLAGS} ${CMAKE_${target_type}_LINKER_FLAGS_${THIS_CONFIG}}")
  message("Project ${target_type}_LINKER_FLAGS: ${TMP_PROJECT_CMAKE_${target_type}_LINKER_FLAGS}")
endforeach(target_type IN ITEMS EXE SHARED)

######################################################################
# CTest
######################################################################
include("${miniqmc_SOURCE_DIR}/CMake/macros.cmake")
set(DROP_METHOD "http")
set(DROP_SITE "cdash.qmcpack.org")
set(DROP_LOCATION "/CDash/submit.php?project=miniQMC")
set(TRIGGER_SITE "")
set(DROP_SITE_CDASH TRUE)
# Increase timeout per test over the default of 1500 seconds (25 minutes)
set(DART_TESTING_TIMEOUT
    3600
    CACHE STRING "Maximum time for one test")
enable_testing()
include(CTest)

if(NOT TEST_MAX_PROCS)
  set(TEST_MAX_PROCS 100)
endif()

######################################################################
message(STATUS "Ready to parse miniQMC source tree")
######################################################################

####################################################################
# add definitions
#  -DH5_USE_16_API for HDF5
####################################################################

add_definitions(-DHAVE_CONFIG_H -DH5_USE_16_API -DADD_)

execute_process(
  COMMAND hostname
  OUTPUT_VARIABLE QMC_HOSTNAME
  OUTPUT_STRIP_TRAILING_WHITESPACE)
configure_file(${miniqmc_SOURCE_DIR}/miniqmc.settings ${miniqmc_BINARY_DIR}/bin/miniqmc.settings)
install(FILES ${miniqmc_BINARY_DIR}/bin/miniqmc.settings DESTINATION bin)

subdirs(src)
