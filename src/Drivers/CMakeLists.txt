SET(ESTEST check_wfc check_determinant)

FOREACH(p ${ESTEST})
  ADD_EXECUTABLE( ${p} ${p}.cpp ${ETEST_KOKKOS_SRC} )
  TARGET_LINK_LIBRARIES(${p} qmcbase qmcutil qmcdevice ${QMC_UTIL_LIBS} ${MPI_LIBRARY})
  IF(QMC_USE_CUDA)
    TARGET_LINK_LIBRARIES(${p} qmc_cuda_params)
  ENDIF(QMC_USE_CUDA)
ENDFOREACH(p ${ESTEST})

#SET(KOKKOS_CHECK_SPO_LIBS "")
# if(QMC_USE_KOKKOS)
#   ADD_LIBRARY(check_spo_kokkos CheckSPOStepsKOKKOS.cpp)
#   SET(KOKKOS_CHECK_SPO_LIBS check_spo_kokkos)
# endif(QMC_USE_KOKKOS)

ADD_EXECUTABLE( check_spo check_spo.cpp CheckSPOSteps.cpp ${ETEST_KOKKOS_SRC} )
TARGET_LINK_LIBRARIES( check_spo qmcbase qmcutil qmcdevice ${QMC_UTIL_LIBS} ${MPI_LIBRARY} ${KOKKOS_CHECK_SPO_LIBS})
IF(QMC_USE_CUDA)
  TARGET_LINK_LIBRARIES( check_spo qmc_cuda_params)
  TARGET_SOURCES(check_spo PUBLIC CheckSPOStepsCUDA.cpp)
ENDIF(QMC_USE_CUDA)
if(QMC_USE_KOKKOS)
  TARGET_SOURCES(check_spo PUBLIC CheckSPOStepsKOKKOS.cpp)
endif(QMC_USE_KOKKOS)


SET(MINIWFS_CUDA_SRC "")
IF(QMC_USE_CUDA)
  SET(MINIWFS_CUDA_SRC ../QMCWaveFunctions/WaveFunctionBuilderCUDA.cpp)
ENDIF(QMC_USE_CUDA)
ADD_LIBRARY(miniwfs ../QMCWaveFunctions/WaveFunction.cpp
  ../QMCWaveFunctions/WaveFunctionBuilder.cpp ${KOKKOS_SRCS} ${MINIWFS_CUDA_SRC})
TARGET_LINK_LIBRARIES(miniwfs qmcutil)
SET(DRIVERS miniqmc) # miniqmc_sync_move)


SET(MINI_FUNCTIONS_SRC Crowd.cpp)
IF(QMC_USE_CUDA)
  SET(MINI_FUNCTIONS_SRC ${MINI_FUNCTIONS_SRC} CrowdCUDA.cpp) # MiniqmcDriverFunctionsCUDA.cpp)
ENDIF(QMC_USE_CUDA)

IF(QMC_USE_KOKKOS)
  SET(MINI_FUNCTIONS_SRC ${MINI_FUNCTIONS_SRC} MiniqmcDriverFunctionsKOKKOS.cpp)
ENDIF(QMC_USE_KOKKOS)

SET(MINI_FUNCTIONS_SRC ${MINI_FUNCTIONS_SRC} MiniqmcDriverFunctions.cpp)
ADD_LIBRARY(mini_functions ${MINI_FUNCTIONS_SRC})
TARGET_LINK_LIBRARIES(mini_functions)

FOREACH(p ${DRIVERS})
  ADD_EXECUTABLE( ${p} MiniqmcOptions.cpp MiniqmcDriver.cpp ${p}.cpp)
  IF(QMC_USE_CUDA)
    TARGET_LINK_LIBRARIES(${p} qmc_cuda_params qmc_numerics)
  ENDIF(QMC_USE_CUDA)
  
  TARGET_LINK_LIBRARIES(${p} qmcutil mini_functions miniwfs qmcbase qmcdevice qmcutil ${QMC_UTIL_LIBS} ${MPI_LIBRARY})
  
ENDFOREACH(p ${ESTEST})



#SET(boost_test exchange_walker)
#FOREACH(p ${boost_test})
#  ADD_EXECUTABLE( ${p}  ${p}.cpp)
#  TARGET_LINK_LIBRARIES(${p} qmcbase qmcutil ${QMC_UTIL_LIBS} boost ${MPI_LIBRARY})
#ENDFOREACH(p ${boost_test})

