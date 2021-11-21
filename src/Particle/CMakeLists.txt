####################################
# create libqmcparticle
####################################
set(PARTICLE VirtualParticleSet.cpp ParticleSet.cpp ParticleSet.BC.cpp DistanceTableAA.cpp DistanceTableAB.cpp
             ParticleSet_builder.cpp)

add_library(qmcparticle ${PARTICLE})
target_link_libraries(qmcparticle PUBLIC qmcutil)

subdirs(tests)
