# Check compiler version
if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 18.0)
  message(FATAL_ERROR "Requires Intel 18.0 or higher ")
endif()

# Enable OpenMP
if(QMC_OMP)
  set(ENABLE_OPENMP 1)
  if(ENABLE_OFFLOAD)
    set(OFFLOAD_TARGET
        "host"
        CACHE STRING "Offload target architecture")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qopenmp -qopenmp-offload=${OFFLOAD_TARGET}")
  else(ENABLE_OFFLOAD)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qopenmp")
  endif(ENABLE_OFFLOAD)
endif(QMC_OMP)

# Suppress compile warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated")

# Set extra optimization specific flags
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -restrict -unroll -ip")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -restrict -unroll -ip")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -restrict -unroll -ip")

# Set prefetch flag
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qopt-prefetch")

#check if -ftz is accepted
check_cxx_compiler_flag("${CMAKE_CXX_FLAGS} -ftz" INTEL_FTZ)
if(INTEL_FTZ)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftz")
endif(INTEL_FTZ)

#------------------------
# Not on Cray's machine
#------------------------
if(NOT CMAKE_SYSTEM_NAME STREQUAL "CrayLinuxEnvironment")

  set(X_OPTION "^-x| -x")
  set(AX_OPTION "^-ax| -ax")
  #check if the user has already specified -x option for cross-compiling.
  if(NOT (CMAKE_CXX_FLAGS MATCHES ${X_OPTION} OR CMAKE_CXX_FLAGS MATCHES ${AX_OPTION}))
    #check if -xHost is accepted
    check_cxx_compiler_flag("-xHost" INTEL_CXX_FLAGS)
    if(INTEL_CXX_FLAGS)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -xHost")
    endif(INTEL_CXX_FLAGS)
  endif() #(CMAKE_CXX_FLAGS MATCHES "-x" OR CMAKE_CXX_FLAGS MATCHES "-ax")

endif()
