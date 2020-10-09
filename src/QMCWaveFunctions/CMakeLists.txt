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

SET(WFS_SRC einspline_spo_omp.cpp
            WaveFunction.cpp SPOSet_builder.cpp
            DiracDeterminant.cpp DiracDeterminantRef.cpp)

IF(USE_OBJECT_TARGET)
  ADD_LIBRARY(qmcwfs OBJECT ${WFS_SRC})
ELSE(USE_OBJECT_TARGET)
  ADD_LIBRARY(qmcwfs ${WFS_SRC})
ENDIF(USE_OBJECT_TARGET)

target_link_libraries(qmcwfs PUBLIC qmcbase)

SUBDIRS(tests)
