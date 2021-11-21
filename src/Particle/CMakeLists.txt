#//////////////////////////////////////////////////////////////////////////////////////
#// This file is distributed under the University of Illinois/NCSA Open Source License.
#// See LICENSE file in top directory for details.
#//
#// Copyright (c) 2021 QMCPACK developers.
#//
#// File developed by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
#//
#// File created by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
#//////////////////////////////////////////////////////////////////////////////////////


set(PARTICLE VirtualParticleSet.cpp ParticleSet.cpp ParticleSet.BC.cpp DistanceTableAA.cpp DistanceTableAB.cpp
             ParticleSet_builder.cpp)

add_library(qmcparticle ${PARTICLE})
target_link_libraries(qmcparticle PUBLIC qmcutil)

subdirs(tests)
