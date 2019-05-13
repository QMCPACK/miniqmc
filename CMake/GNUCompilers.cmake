# Check compiler version
IF ( CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.8 )
MESSAGE(FATAL_ERROR "Requires gcc 4.8 or higher ")
ENDIF()

# Enable OpenMP
IF(QMC_OMP)
  SET(ENABLE_OPENMP 1)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
ENDIF(QMC_OMP)

# Set gnu specfic flags (which we always want)
ADD_DEFINITIONS( -Drestrict=__restrict__ )

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fomit-frame-pointer -finline-limit=1000 -fstrict-aliasing -funroll-all-loops -D__forceinline=inline")

# Suppress compile warnings
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated")

# Set extra optimization specific flags
SET( CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -ffast-math" )
SET( CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -ffast-math" )

#--------------------------------------
# Special architectural flags
#--------------------------------------
# case arch
#     x86_64: -march
#     powerpc: -mpcu
#     arm: -mpcu
#     default or cray: none
#--------------------------------------
IF((NOT $ENV{CRAYPE_VERSION} MATCHES "."))
  # It's a cray machine. Don't do anything
ELSEIF(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
  # the case for x86_64
  #check if the user has already specified -march=XXXX option for cross-compiling.
  if(NOT CMAKE_CXX_FLAGS MATCHES "-march=")
    # use -march=native
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
  endif(NOT CMAKE_CXX_FLAGS MATCHES "-march=")
ELSEIF(CMAKE_SYSTEM_PROCESSOR MATCHES "ppc64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  # the case for PowerPC and ARM
  #check if the user has already specified -mcpu=XXXX option for cross-compiling.
  if(NOT CMAKE_CXX_FLAGS MATCHES "-mcpu=")
    # use -mcpu=native
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=native")
  endif(NOT CMAKE_CXX_FLAGS MATCHES "-mcpu=")
ENDIF()

# Add static flags if necessary
IF(QMC_BUILD_STATIC)
SET(CMAKE_CXX_LINK_FLAGS " -static")
ENDIF(QMC_BUILD_STATIC)

# Coverage
IF (ENABLE_GCOV)
  SET(GCOV_SUPPORTED TRUE)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
  SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
  SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} --coverage")
ENDIF(ENABLE_GCOV)


