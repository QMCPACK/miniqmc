# Check compiler version
IF ( CMAKE_CXX_COMPILER_VERSION VERSION_LESS 3.3 )
  MESSAGE(STATUS "Compiler Version ${CMAKE_CXX_COMPILER_VERSION}")
  MESSAGE(FATAL_ERROR "Requires clang 3.3 or higher ")
ENDIF()

# Enable OpenMP
# #IF(NOT QMC_USE_KOKKOS)
IF(QMC_OMP)
  SET(ENABLE_OPENMP 1)
  IF(ENABLE_OFFLOAD)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda")
  ELSE()
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
  ENDIF()
ENDIF(QMC_OMP)
#ENDIF()

# IF(QMC_USE_KOKKOS)
#   SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --expt-relaxed-constexpr")
# ENDIF(QMC_USE_KOKKOS)


SET(QMC_USE_CLANG_CUDA FALSE CACHE BOOL "Use clang to build CUDA code")
# IF(QMC_USE_CUDA AND QMC_CLANG_CUDA)
  
#  add_definitions( -D__CUDACC__)
# ENDIF(QMC_USE_CUDA)

# Set clang specfic flags (which we always want)
ADD_DEFINITIONS( -Drestrict=__restrict__ )

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__forceinline=inline")
SET( HAVE_POSIX_MEMALIGN 0 )    # Clang doesn't support -malign-double

# Suppress compile warnings
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated -Wno-unused-value -Wno-ignored-attributes")
IF ( CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 3.8.0 )
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ") #-Wno-undefined-var-template
ENDIF()

# Set extra optimization specific flags
SET( CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fomit-frame-pointer -fstrict-aliasing -ffast-math" )
SET( CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -ffast-math" )
SET( CMAKE_C_FLAGS_DEBUG     "${CMAKE_C_FLAGS_DEBUG} -fno-omit-frame-pointer -fstandalone-debug" )
SET( CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}  -DBOOST_HANA_CONFIG_ENABLE_DEBUG_MODE -fno-omit-frame-pointer -fstandalone-debug" ) #-Wno-undefined-var-template

#--------------------------------------
# Neither on Cray's machine nor PowerPC
#--------------------------------------
IF((NOT $ENV{CRAYPE_VERSION} MATCHES ".") AND (NOT CMAKE_SYSTEM_PROCESSOR MATCHES "ppc64"))

#check if the user has already specified -march=XXXX option for cross-compiling.
if(CMAKE_CXX_FLAGS MATCHES "-march=")
else() #(CMAKE_CXX_FLAGS MATCHES "-march=")
  # use -march=native
  if(NOT ENABLE_OFFLOAD)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
  endif()
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

SET(XRAY_PROFILE FALSE CACHE BOOL "Use llvm xray profiling")
SET(XRAY_INSTRUCTION_THRESHOLD 200 CACHE INT "Instruction threshold for xray instrumentation")
MARK_AS_ADVANCED(XRAY_PROFILE)
MARK_AS_ADVANCED(XRAY_INSTRUCTION_THRESHOLD)

IF(XRAY_PROFILE)
  set(XRAY_FLAGS "-fxray-instrument -fxray-instruction-threshold=${XRAY_INSTRUCTION_THRESHOLD}")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${XRAY_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${XRAY_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${XRAY_FLAGS}")
ENDIF(XRAY_PROFILE)

SET(LLVM_SANITIZE_ADDRESS FALSE CACHE BOOL "Use llvm sanitize address library")
MARK_AS_ADVANCED(LLVM_SANITIZE_ADDRESS)
IF(LLVM_SANITIZE_ADDRESS)
  SET(CMAKE_C_FLAGS_DEBUG "-fsanitize=address -fsanitize=address -fsanitize-address-use-after-scope ${CMAKE_C_FLAGS_DEBUG}")
  SET(CMAKE_CXX_FLAGS_DEBUG "-fsanitize=address -fsanitize-address-use-after-scope ${CMAKE_CXX_FLAGS_DEBUG}")
  SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address -fsanitize-address-use-after-scope")
  SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=address -fsanitize-address-use-after-scope")
ENDIF(LLVM_SANITIZE_ADDRESS)

SET(LLVM_SANITIZE_MEMORY FALSE CACHE BOOL "Use llvm sanitize memory library")
MARK_AS_ADVANCED(LLVM_SANITIZE_MEMORY)
IF(LLVM_SANITIZE_MEMORY)
  SET(LLVM_BLACKLIST_SANITIZE_MEMORY "-fsanitize-blacklist=${PROJECT_SOURCE_DIR}/llvm_misc/memory_sanitizer_blacklist.txt")
  SET(CMAKE_C_FLAGS_DEBUG "-fsanitize=memory ${LLVM_BLACKLIST_SANITIZE_MEMORY} ${CMAKE_C_FLAGS_DEBUG}")
  SET(CMAKE_CXX_FLAGS_DEBUG "-fsanitize=memory ${LLVM_BLACKLIST_SANITIZE_MEMORY} ${CMAKE_CXX_FLAGS_DEBUG}")
  SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=memory ${LLVM_BLACKLIST_SANITIZE_MEMORY}")
  SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=memory ${LLVM_BLACKLIST_SANITIZE_MEMORY}")
ENDIF(LLVM_SANITIZE_MEMORY)
