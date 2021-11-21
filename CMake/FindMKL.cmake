# Simple file to refine MKL search.
# It relies on FindLAPACK to locate MKL library files and set up linker options first.
include(CheckCXXSourceCompiles)

set(MKL_LIBRARIES ${LAPACK_LINKER_FLAGS} ${LAPACK_LIBRARIES})

message(STATUS "Looking for Intel MKL library header files")

# Finding and setting the MKL_INCLUDE_DIRECTORIES based on MKL_ROOT, $ENV{MKLROOT}, $ENV{MKL_ROOT}, $ENV{MKL_HOME}
# Extremely Basic Support of common mkl module environment variables
find_path(
  MKL_INCLUDE_DIRECTORIES "mkl.h"
  HINTS ${MKL_ROOT} $ENV{MKLROOT} $ENV{MKL_ROOT} $ENV{MKL_HOME}
  PATH_SUFFIXES include)
if(NOT MKL_INCLUDE_DIRECTORIES)
  # Finding MKL headers in the system
  find_path(MKL_INCLUDE_DIRECTORIES "mkl.h" PATH_SUFFIXES mkl)
endif()

if(MKL_INCLUDE_DIRECTORIES)
  message(STATUS "MKL_INCLUDE_DIRECTORIES: ${MKL_INCLUDE_DIRECTORIES}")
else(MKL_INCLUDE_DIRECTORIES)
  message(STATUS "mkl.h cannot be found")
  if(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    message(
      FATAL_ERROR
        "Intel's standard compilervar.sh sets the env variable MKLROOT.\n"
        "If you are invoking icc without the customary environment\n"
        "you must set the the environment variable or pass cmake MKL_ROOT.")
  else(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    message(FATAL_ERROR "Pass mkl root directory to cmake via MKL_ROOT.")
  endif(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
endif(MKL_INCLUDE_DIRECTORIES)

# Check for mkl.h
file(WRITE "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/src_mkl.cxx"
     "#include <iostream>\n #include <mkl.h>\n int main() { return 0; }\n")
try_compile(
  HAVE_MKL ${CMAKE_BINARY_DIR} ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/src_mkl.cxx
  CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${MKL_INCLUDE_DIRECTORIES} "
  LINK_LIBRARIES "${MKL_LIBRARIES}"
  OUTPUT_VARIABLE MKL_OUT)
if(NOT HAVE_MKL)
  message("${MKL_OUT}")
endif(NOT HAVE_MKL)

if(HAVE_MKL)
  set(MKL_FOUND TRUE)
  message(STATUS "MKL found: HAVE_MKL=${HAVE_MKL}")

  #Add BLAS_LAPACK header
  set_target_properties(Math::BLAS_LAPACK PROPERTIES INTERFACE_COMPILE_DEFINITIONS "HAVE_MKL"
                                                     INTERFACE_INCLUDE_DIRECTORIES "${MKL_INCLUDE_DIRECTORIES}")

else(HAVE_MKL)
  set(MKL_FOUND FALSE)
  message(STATUS "MKL header files not found")
endif(HAVE_MKL)
