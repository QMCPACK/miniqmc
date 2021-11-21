#//////////////////////////////////////////////////////////////////////////////////////
#// This file is distributed under the University of Illinois/NCSA Open Source License.
#// See LICENSE file in top directory for details.
#//
#// Copyright (c) 2019 QMCPACK developers.
#//
#// File developed by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
#//
#// File created by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
#//////////////////////////////////////////////////////////////////////////////////////

set(WFS_SRC einspline_spo_omp.cpp WaveFunction.cpp SPOSet_builder.cpp DiracDeterminant.cpp DiracDeterminantRef.cpp)

if(USE_OBJECT_TARGET)
  add_library(qmcwfs OBJECT ${WFS_SRC})
else(USE_OBJECT_TARGET)
  add_library(qmcwfs ${WFS_SRC})
endif(USE_OBJECT_TARGET)

target_link_libraries(qmcwfs PUBLIC qmcparticle)

target_link_libraries(qmcwfs PRIVATE platform_LA)

subdirs(tests)
