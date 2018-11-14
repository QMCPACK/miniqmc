if(QMC_BUILD_LEVEL GREATER 4)
# add apps XYZ.cpp, e.g., qmc_particles.cpp
#SET(ESTEST einspline_smp einspline_spo qmc_particles moveonsphere twobody ptclset)
SET(ESTEST check_wfc check_spo check_determinant)

FOREACH(p ${ESTEST})
  ADD_EXECUTABLE( ${p}  ${p}.cpp ${ETEST_KOKKOS_SRC})
  TARGET_LINK_LIBRARIES(${p} qmcbase qmcutil ${QMC_UTIL_LIBS} ${MPI_LIBRARY})
ENDFOREACH(p ${ESTEST})

# SET(THIS_TEST check_determinant)
# IF(QMC_USE_KOKKOS)
#   SET(ETEST_KOKKOS_SRC "test/CheckDeterminantHelpersKOKKOS.cpp")
# ENDIF(QMC_USE_KOKKOS)
# ADD_EXECUTABLE( ${THIS_TEST}  ${THIS_TEST}.cpp ${ETEST_KOKKOS_SRC})
# TARGET_LINK_LIBRARIES(${THIS_TEST} qmcbase qmcutil ${QMC_UTIL_LIBS} ${MPI_LIBRARY})

ADD_LIBRARY(miniwfs ../QMCWaveFunctions/WaveFunction.cpp ../QMCWaveFunctions/SPOSet_builder.cpp)

SET(DRIVERS miniqmc miniqmc_sync_move)

FOREACH(p ${DRIVERS})
  ADD_EXECUTABLE( ${p}  ${p}.cpp)
  TARGET_LINK_LIBRARIES(${p} miniwfs qmcbase qmcutil ${QMC_UTIL_LIBS} ${MPI_LIBRARY})
ENDFOREACH(p ${ESTEST})

endif()

#SET(boost_test exchange_walker)
#FOREACH(p ${boost_test})
#  ADD_EXECUTABLE( ${p}  ${p}.cpp)
#  TARGET_LINK_LIBRARIES(${p} qmcbase qmcutil ${QMC_UTIL_LIBS} boost ${MPI_LIBRARY})
#ENDFOREACH(p ${boost_test})

