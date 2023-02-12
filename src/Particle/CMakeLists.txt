#//////////////////////////////////////////////////////////////////////////////////////
#// This file is distributed under the University of Illinois/NCSA Open Source License.
#// See LICENSE file in top directory for details.
#//
#// Copyright (c) 2020 QMCPACK developers.
#//
#// File developed by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
#//
#// File created by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
#//////////////////////////////////////////////////////////////////////////////////////

####################################
# create libqmcparticle
####################################
set(PARTICLE
    createDistanceTableAA.cpp   ParticleSet_builder.cpp        createDistanceTableAB.cpp           DynamicCoordinatesBuilder.cpp  SpeciesSet.cpp
    MCCoords.cpp                   ParticleSet.cpp     VirtualParticleSet.cpp)

if(USE_OBJECT_TARGET)
  add_library(qmcparticle OBJECT ${PARTICLE})
else(USE_OBJECT_TARGET)
  add_library(qmcparticle ${PARTICLE})
endif(USE_OBJECT_TARGET)

target_include_directories(qmcparticle PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(qmcparticle PRIVATE platform_cpu_LA)
target_link_libraries(qmcparticle PUBLIC qmcutil platform_runtime)
set(PARTICLE_OMPTARGET_SRCS
    createDistanceTableAAOMPTarget.cpp
    createDistanceTableABOMPTarget.cpp)

if(USE_OBJECT_TARGET)
  add_library(qmcparticle_omptarget OBJECT ${PARTICLE_OMPTARGET_SRCS})
else(USE_OBJECT_TARGET)
  add_library(qmcparticle_omptarget ${PARTICLE_OMPTARGET_SRCS})
endif(USE_OBJECT_TARGET)

target_include_directories(qmcparticle_omptarget PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(qmcparticle_omptarget PUBLIC qmcutil)

target_compile_options(qmcparticle_omptarget PRIVATE "$<$<BOOL:${ENABLE_OFFLOAD_CLANG_DEBUG_O3}>:$<$<CONFIG:DEBUG>:-O3>>")

target_link_libraries(qmcparticle PUBLIC qmcparticle_omptarget)

add_subdirectory(tests)
