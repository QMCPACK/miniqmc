if(QMC_BUILD_LEVEL GREATER 4)
# add apps XYZ.cpp, e.g., qmc_particles.cpp
#SET(ESTEST einspline_smp einspline_spo qmc_particles moveonsphere twobody ptclset)
SET(ESTEST check_wfc check_spo check_determinant)

FOREACH(p ${ESTEST})
  ADD_EXECUTABLE( ${p}  ${p}.cpp)
  TARGET_LINK_LIBRARIES(${p} qmcbase qmcutil ${QMC_UTIL_LIBS} ${MPI_LIBRARY})
ENDFOREACH(p ${ESTEST})

ADD_LIBRARY(miniwfs ../QMCWaveFunctions/WaveFunction.cpp ../QMCWaveFunctions/SPOSet_builder.cpp)

SET(DRIVERS miniqmc miniqmc_sync_move)

FOREACH(p ${DRIVERS})
  ADD_EXECUTABLE( ${p}  ${p}.cpp)
  TARGET_LINK_LIBRARIES(${p} miniwfs qmcbase qmcutil ${QMC_UTIL_LIBS} ${MPI_LIBRARY})
ENDFOREACH(p ${ESTEST})

SET(UNIT_TEST_SRCS ../Particle/tests/test_bconds.cpp)
ADD_EXECUTABLE(unit_test_miniqmc ${UNIT_TEST_SRCS})
TARGET_LINK_LIBRARIES(unit_test_miniqmc miniwfs qmcbase qmcutil ${QMC_UTIL_LIBS} ${MPI_LIBRARY})
ADD_TEST(NAME unit_test COMMAND unit_test_miniqmc)
SET_TESTS_PROPERTIES(unit_test PROPERTIES LABELS "unit")

endif()

#SET(boost_test exchange_walker)
#FOREACH(p ${boost_test})
#  ADD_EXECUTABLE( ${p}  ${p}.cpp)
#  TARGET_LINK_LIBRARIES(${p} qmcbase qmcutil ${QMC_UTIL_LIBS} boost ${MPI_LIBRARY})
#ENDFOREACH(p ${boost_test})

