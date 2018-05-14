if(QMC_BUILD_LEVEL GREATER 4)
# add apps XYZ.cpp, e.g., qmc_particles.cpp
#SET(ESTEST einspline_smp einspline_spo qmc_particles moveonsphere twobody ptclset)
SET(ESTEST check_wfc check_spo check_determinant)

FOREACH(p ${ESTEST})
  ADD_EXECUTABLE( ${p}  ${p}.cpp)
  TARGET_LINK_LIBRARIES(${p} qmcbase qmcutil ${QMC_UTIL_LIBS} ${MPI_LIBRARY})
ENDFOREACH(p ${ESTEST})

ADD_LIBRARY(miniwfs ../QMCWaveFunctions/WaveFunctionRef.cpp ../QMCWaveFunctions/WaveFunction.cpp)

ADD_EXECUTABLE(miniqmc miniqmc.cpp)
TARGET_LINK_LIBRARIES(miniqmc miniwfs qmcbase qmcutil ${QMC_UTIL_LIBS} ${MPI_LIBRARY})

#ADD_EXECUTABLE(diff_wfs diff_wfs.cpp)
#TARGET_LINK_LIBRARIES(diff_wfs miniwfs qmcbase qmcutil ${QMC_UTIL_LIBS} ${MPI_LIBRARY})

endif()

#SET(boost_test exchange_walker)
#FOREACH(p ${boost_test})
#  ADD_EXECUTABLE( ${p}  ${p}.cpp)
#  TARGET_LINK_LIBRARIES(${p} qmcbase qmcutil ${QMC_UTIL_LIBS} boost ${MPI_LIBRARY})
#ENDFOREACH(p ${boost_test})

