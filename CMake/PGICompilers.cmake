# Enable OpenMP
# If just -mp is specified, OMP_NUM_THREADS must be set in order to run in parallel
# Specifying 'allcores' will run on all cores if OMP_NUM_THREADS is not set (which seems
#  to be the default for other OpenMP implementations)
if(QMC_OMP)
  set(ENABLE_OPENMP 1)
  if(ENABLE_OFFLOAD AND NOT CMAKE_SYSTEM_NAME STREQUAL "CrayLinuxEnvironment")
    if(NOT DEFINED OFFLOAD_ARCH)
      message(FATAL_ERROR "NVIDIA HPC compiler requires -gpu=ccXX option set based on the target GPU architecture! "
                          "Please add -DOFFLOAD_ARCH=ccXX to cmake. For example, cc70 is for Volta.")
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mp=gpu")
    set(OPENMP_OFFLOAD_COMPILE_OPTIONS "-gpu=${OFFLOAD_ARCH}")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mp=allcores")
  endif()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__NO_UDR")
endif(QMC_OMP)

add_definitions(-Drestrict=__restrict__)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__forceinline=inline")

# Set extra optimization specific flags
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fast")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -fast")

# Setting this to 'OFF' adds the -A flag, which enforces strict standard compliance
#  and causes the compilation to fail with some GNU header files
set(CMAKE_CXX_EXTENSIONS ON)

# Add static flags if necessary
if(QMC_BUILD_STATIC)
  set(CMAKE_CXX_LINK_FLAGS " -Bstatic")
endif(QMC_BUILD_STATIC)
