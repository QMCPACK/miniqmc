# Check compiler version
IF ( CMAKE_CXX_COMPILER_VERSION VERSION_LESS 3.3 )
  MESSAGE(STATUS "Compiler Version ${CMAKE_CXX_COMPILER_VERSION}")
  MESSAGE(FATAL_ERROR "Requires clang 3.3 or higher ")
ENDIF()

# Enable OpenMP
IF(QMC_OMP)
  SET(ENABLE_OPENMP 1)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
ENDIF(QMC_OMP)

# Set clang specfic flags (which we always want)
ADD_DEFINITIONS( -Drestrict=__restrict__ )

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fomit-frame-pointer -fstrict-aliasing -D__forceinline=inline")
SET( HAVE_POSIX_MEMALIGN 0 )    # Clang doesn't support -malign-double

# Suppress compile warnings
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated -Wno-unused-value")
IF ( CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 3.8 )
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-undefined-var-template")
ENDIF()

# Set extra optimization specific flags
SET( CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -ffast-math" )
SET( CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -ffast-math" )

#--------------------------------------
# Neither on Cray's machine nor PowerPC
#--------------------------------------
IF((NOT $ENV{CRAYPE_VERSION} MATCHES ".") AND (NOT CMAKE_SYSTEM_PROCESSOR MATCHES "ppc64"))

#check if the user has already specified -march=XXXX option for cross-compiling.
if(CMAKE_CXX_FLAGS MATCHES "-march=")
else() #(CMAKE_CXX_FLAGS MATCHES "-march=")
  # use -march=native
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif() #(CMAKE_CXX_FLAGS MATCHES "-march=")

ENDIF((NOT $ENV{CRAYPE_VERSION} MATCHES ".") AND (NOT CMAKE_SYSTEM_PROCESSOR MATCHES "ppc64"))

# Add static flags if necessary
IF(QMC_BUILD_STATIC)
    SET(CMAKE_CXX_LINK_FLAGS " -static")
ENDIF(QMC_BUILD_STATIC)

# Coverage
IF (ENABLE_GCOV)
  SET(GCOV_COVERAGE TRUE)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
  SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
  SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} --coverage")
ENDIF(ENABLE_GCOV)

