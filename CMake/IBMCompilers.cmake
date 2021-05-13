IF(CMAKE_SYSTEM_PROCESSOR MATCHES "ppc64le")
MESSAGE(STATUS "Power8+ system using xlC/xlc/xlf")

ADD_DEFINITIONS( -Drestrict=__restrict__ )

# Clean up flags
IF(CMAKE_CXX_FLAGS MATCHES "-qhalt=e")
  SET(CMAKE_CXX_FLAGS "")
ENDIF()

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__forceinline=inline")

# Suppress compile warnings
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated -Wno-unused-value")

# Set extra optimization specific flags
SET( CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG" )
SET( CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g -O3" )

# Set language standardards
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qnoxlcompatmacros")

# CMake < 3.9 didn't have support for language standards in XL
IF(NOT CMAKE_CXX11_STANDARD_COMPILE_OPTION)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
ENDIF()

IF(QMC_OMP)
  SET(ENABLE_OPENMP 1)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qsmp=omp")
  IF(ENABLE_OFFLOAD)
    set(OPENMP_OFFLOAD_COMPILE_OPTIONS "-qoffload")
  ENDIF(ENABLE_OFFLOAD)
ELSE(QMC_OMP)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qnothreaded")
ENDIF(QMC_OMP)

# Add static flags if necessary
IF(QMC_BUILD_STATIC)
    SET(CMAKE_CXX_LINK_FLAGS " -static")
ENDIF(QMC_BUILD_STATIC)

ENDIF(CMAKE_SYSTEM_PROCESSOR MATCHES "ppc64le")
