if(QMC_BUILD_LEVEL GREATER 4)
# add apps XYZ.cpp, e.g., qmc_particles.cpp
#SET(ESTEST einspline_smp einspline_spo qmc_particles moveonsphere twobody ptclset)
SET(ESTEST check_wfc check_spo omp4offload check_determinant)

FOREACH(p ${ESTEST})
  ADD_EXECUTABLE( ${p}  ${p}.cpp)
  TARGET_LINK_LIBRARIES(${p} qmcbase qmcutil ${QMC_UTIL_LIBS} ${MPI_LIBRARY})
ENDFOREACH(p ${ESTEST})

SET(DRIVERS check_spo_batched miniqmc miniqmc_sync_move)

FOREACH(p ${DRIVERS})
  ADD_EXECUTABLE( ${p}  ${p}.cpp)
  TARGET_LINK_LIBRARIES(${p} qmcwfs qmcbase qmcutil ${QMC_UTIL_LIBS} ${MPI_LIBRARY})
ENDFOREACH(p ${ESTEST})

endif()

#SET(boost_test exchange_walker)
#FOREACH(p ${boost_test})
#  ADD_EXECUTABLE( ${p}  ${p}.cpp)
#  TARGET_LINK_LIBRARIES(${p} qmcbase qmcutil ${QMC_UTIL_LIBS} boost ${MPI_LIBRARY})
#ENDFOREACH(p ${boost_test})

