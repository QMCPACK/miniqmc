  ####################################
  # create libqmcparticle 
  ####################################
  SET(PARTICLE
    VirtualParticleSet.cpp
    ParticleSet.cpp
    ParticleSet.BC.cpp
    DistanceTableAA.cpp
    DistanceTableAB.cpp
    ParticleSet_builder.cpp
    )

  ADD_LIBRARY(qmcparticle ${PARTICLE})
  TARGET_LINK_LIBRARIES(qmcparticle PUBLIC qmcutil)

  SUBDIRS(tests)
