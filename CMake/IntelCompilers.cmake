# Check compiler version
SET(INTEL_COMPILER 1)
IF ( CMAKE_CXX_COMPILER_VERSION VERSION_LESS 18.0 )
MESSAGE(FATAL_ERROR "Requires Intel 18.0 or higher ")
ENDIF()

# Enable OpenMP
IF(QMC_OMP)
  SET(ENABLE_OPENMP 1)
  IF(ENABLE_OFFLOAD)
    SET(OFFLOAD_TARGET "host" CACHE STRING "Offload target architecture")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qopenmp -qopenmp-offload=${OFFLOAD_TARGET}")
  ELSE(ENABLE_OFFLOAD)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qopenmp")
  ENDIF(ENABLE_OFFLOAD)
ENDIF(QMC_OMP)

# Suppress compile warnings
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated")

# Set extra optimization specific flags
SET( CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -restrict -unroll -ip" )
SET( CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -restrict -unroll -ip" )
SET( CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -restrict -unroll -ip" )

# Set prefetch flag
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qopt-prefetch" )

#check if -ftz is accepted
CHECK_CXX_COMPILER_FLAG( "${CMAKE_CXX_FLAGS} -ftz" INTEL_FTZ )
IF( INTEL_FTZ)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftz" )
ENDIF( INTEL_FTZ)

#------------------------
# Not on Cray's machine
#------------------------
IF(NOT CMAKE_SYSTEM_NAME STREQUAL "CrayLinuxEnvironment")

SET(X_OPTION "^-x| -x")
SET(AX_OPTION "^-ax| -ax")
#check if the user has already specified -x option for cross-compiling.
if(NOT (CMAKE_CXX_FLAGS MATCHES ${X_OPTION} OR CMAKE_CXX_FLAGS MATCHES ${AX_OPTION}))
  #check if -xHost is accepted
  CHECK_CXX_COMPILER_FLAG( "-xHost" INTEL_CXX_FLAGS )
  IF(INTEL_CXX_FLAGS)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -xHost")
  ENDIF(INTEL_CXX_FLAGS)
endif() #(CMAKE_CXX_FLAGS MATCHES "-x" OR CMAKE_CXX_FLAGS MATCHES "-ax")

ENDIF()
