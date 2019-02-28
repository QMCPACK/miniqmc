if(QMC_BUILD_LEVEL GREATER 4)
# add apps XYZ.cpp, e.g., qmc_particles.cpp
#SET(ESTEST einspline_smp EinsplineSPO qmc_particles moveonsphere twobody ptclset)

SET(ESTEST check_wfc check_spo check_determinant)

FOREACH(p ${ESTEST})
  ADD_EXECUTABLE( ${p} ${p}.cpp ${ETEST_KOKKOS_SRC} )
  TARGET_LINK_LIBRARIES(${p} qmcbase qmcutil qmcdevice ${QMC_UTIL_LIBS} ${MPI_LIBRARY})
  IF(QMC_USE_CUDA)
    TARGET_LINK_LIBRARIES(${p} qmc_cuda_params)
  ENDIF(QMC_USE_CUDA)
ENDFOREACH(p ${ESTEST})

# SET(THIS_TEST check_determinant)
# IF(QMC_USE_KOKKOS)
#   SET(ETEST_KOKKOS_SRC "test/CheckDeterminantStepsKOKKOS.cpp")
# ENDIF(QMC_USE_KOKKOS)
# ADD_EXECUTABLE( ${THIS_TEST}  ${THIS_TEST}.cpp ${ETEST_KOKKOS_SRC})
# TARGET_LINK_LIBRARIES(${THIS_TEST} qmcbase qmcutil ${QMC_UTIL_LIBS} ${MPI_LIBRARY})

ADD_LIBRARY(miniwfs ../QMCWaveFunctions/WaveFunction.cpp
  ../QMCWaveFunctions/WaveFunctionBuilder.cpp ${KOKKOS_SRCS})
TARGET_LINK_LIBRARIES(miniwfs qmcutil)
SET(DRIVERS miniqmc) # miniqmc_sync_move)


SET(MINI_FUNCTIONS_SRC Movers.cpp MiniqmcDriverFunctions.cpp)
IF(QMC_USE_KOKKOS)
  SET(MINI_FUNCTIONS_SRC ${MINI_FUNCTIONS_SRC} MiniqmcDriverFunctionsKokkos.cpp)
ENDIF(QMC_USE_KOKKOS)
ADD_LIBRARY(mini_functions ${MINI_FUNCTIONS_SRC})
TARGET_LINK_LIBRARIES(mini_functions)

FOREACH(p ${DRIVERS})
  ADD_EXECUTABLE( ${p} MiniqmcOptions.cpp MiniqmcDriver.cpp ${p}.cpp)
  IF(QMC_USE_CUDA)
    TARGET_LINK_LIBRARIES(${p} qmc_cuda_params qmc_numerics)
  ENDIF(QMC_USE_CUDA)
  
  TARGET_LINK_LIBRARIES(${p} qmcutil mini_functions miniwfs qmcbase qmcdevice qmcutil ${QMC_UTIL_LIBS} ${MPI_LIBRARY})
  
ENDFOREACH(p ${ESTEST})

endif()

#SET(boost_test exchange_walker)
#FOREACH(p ${boost_test})
#  ADD_EXECUTABLE( ${p}  ${p}.cpp)
#  TARGET_LINK_LIBRARIES(${p} qmcbase qmcutil ${QMC_UTIL_LIBS} boost ${MPI_LIBRARY})
#ENDFOREACH(p ${boost_test})

