# Check compiler version
if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.8)
  message(FATAL_ERROR "Requires gcc 4.8 or higher ")
endif()

# Enable OpenMP
if(QMC_OMP)
  set(ENABLE_OPENMP 1)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
  if(ENABLE_OFFLOAD AND NOT CMAKE_SYSTEM_NAME STREQUAL "CrayLinuxEnvironment")
    set(OFFLOAD_TARGET
        "nvptx-none"
        CACHE STRING "Offload target architecture")
    set(OPENMP_OFFLOAD_COMPILE_OPTIONS "-foffload=${OFFLOAD_TARGET} -foffload=\"-lm -latomic\"")
  endif()
endif(QMC_OMP)

# Set gnu specfic flags (which we always want)
add_definitions(-Drestrict=__restrict__)

set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -fomit-frame-pointer -finline-limit=1000 -fstrict-aliasing -funroll-all-loops -D__forceinline=inline"
)

# Suppress compile warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated")

# Set extra optimization specific flags
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -ffast-math")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -ffast-math")

#--------------------------------------
# Special architectural flags
#--------------------------------------
# case arch
#     x86_64: -march
#     powerpc: -mpcu
#     arm: -mpcu
#     default or cray: none
#--------------------------------------
if(CMAKE_SYSTEM_NAME STREQUAL "CrayLinuxEnvironment")
  # It's a cray machine. Don't do anything
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
  # the case for x86_64
  #check if the user has already specified -march=XXXX option for cross-compiling.
  if(NOT CMAKE_CXX_FLAGS MATCHES "-march=")
    # use -march=native
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
  endif(NOT CMAKE_CXX_FLAGS MATCHES "-march=")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "ppc64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  # the case for PowerPC and ARM
  #check if the user has already specified -mcpu=XXXX option for cross-compiling.
  if(NOT CMAKE_CXX_FLAGS MATCHES "-mcpu=")
    # use -mcpu=native
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=native")
  endif(NOT CMAKE_CXX_FLAGS MATCHES "-mcpu=")
endif()

# Add static flags if necessary
if(QMC_BUILD_STATIC)
  set(CMAKE_CXX_LINK_FLAGS " -static")
endif(QMC_BUILD_STATIC)

# Coverage
if(ENABLE_GCOV)
  set(GCOV_SUPPORTED TRUE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} --coverage")
endif(ENABLE_GCOV)
