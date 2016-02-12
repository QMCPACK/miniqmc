# Check compiler version
IF ( CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.4 )
MESSAGE(FATAL_ERROR "Require gcc 4.4 or higher ")
ENDIF()

# Enable OpenMP
SET(ENABLE_OPENMP 1)
SET(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} -fopenmp")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

# Set the std
SET(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} -std=c99")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

# Set gnu specfic flags (which we always want)
ADD_DEFINITIONS( -Drestrict=__restrict__ )
ADD_DEFINITIONS( -DADD_ )
ADD_DEFINITIONS( -DINLINE_ALL=inline )
SET(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -malign-double -fomit-frame-pointer -finline-limit=1000 -fstrict-aliasing -funroll-all-loops")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -malign-double -fomit-frame-pointer -finline-limit=1000 -fstrict-aliasing -funroll-all-loops")

# Suppress compile warnings
SET(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -Wno-deprecated")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated")

# Set extra optimization specific flags
SET( CMAKE_C_FLAGS_RELEASE   "${CMAKE_C_FLAGS_RELEASE}   -ffast-math" )
SET( CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -ffast-math" )
SET( CMAKE_C_FLAGS_RELWITHDEBINFO   "${CMAKE_C_FLAGS_RELWITHDEBINFO}   -ffast-math" )
SET( CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -ffast-math" )

# Enable mmx/sse instructions if posix_memalign exists
IF(HAVE_POSIX_MEMALIGN)

# Check for mmx flags
SET(CMAKE_TRY_GNU_CC_FLAGS "-mmmx")
CHECK_C_COMPILER_FLAG(${CMAKE_TRY_GNU_CC_FLAGS} GNU_CC_FLAGS)
IF(GNU_CC_FLAGS)
  SET(HAVE_MMX 1)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mmmx")
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mmmx")
ENDIF(GNU_CC_FLAGS)

# Check for msse flags
SET(CMAKE_TRY_GNU_CC_FLAGS "-msse")
CHECK_C_COMPILER_FLAG(${CMAKE_TRY_GNU_CC_FLAGS} GNU_CC_FLAGS)
IF(GNU_CC_FLAGS)
  SET(HAVE_SSE 1)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse")
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -msse")
ENDIF(GNU_CC_FLAGS)

# Check for msse2 flags
SET(CMAKE_TRY_GNU_CXX_FLAGS "-msse2")
CHECK_C_COMPILER_FLAG(${CMAKE_TRY_GNU_CC_FLAGS} GNU_CC_FLAGS)
IF(GNU_CC_FLAGS)
  SET(HAVE_SSE2 1)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse2")
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -msse2")
ENDIF(GNU_CC_FLAGS)

# Check for msse3 flags
SET(CMAKE_TRY_GNU_CC_FLAGS "-msse3")
CHECK_C_COMPILER_FLAG(${CMAKE_TRY_GNU_CC_FLAGS} GNU_CC_FLAGS)
IF(GNU_CC_FLAGS)
  SET(HAVE_SSE3 1)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse3")
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -msse3")
ENDIF(GNU_CC_FLAGS)

# Check for msse4.1 flags
SET(CMAKE_TRY_GNU_CC_FLAGS "-msse4.1")
CHECK_C_COMPILER_FLAG(${CMAKE_TRY_GNU_CC_FLAGS} GNU_CC_FLAGS)
IF(GNU_CC_FLAGS)
  SET(HAVE_SSE41 1)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.1")
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -msse4.1")
ENDIF(GNU_CC_FLAGS)

ENDIF(HAVE_POSIX_MEMALIGN)

# Add static flags if necessary
IF(QMC_BUILD_STATIC)
SET(CMAKE_CXX_LINK_FLAGS " -static")
ENDIF(QMC_BUILD_STATIC)

# Add enviornmental flags
SET(CMAKE_CXX_FLAGS "$ENV{CXX_FLAGS} ${CMAKE_CXX_FLAGS}")
SET(CMAKE_C_FLAGS "$ENV{CC_FLAGS} ${CMAKE_C_FLAGS}")

