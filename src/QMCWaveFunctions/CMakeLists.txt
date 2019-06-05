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

ADD_LIBRARY(qmcwfs
            ../QMCWaveFunctions/WaveFunction.cpp ../QMCWaveFunctions/SPOSet_builder.cpp
            ../QMCWaveFunctions/DiracDeterminant.cpp ../QMCWaveFunctions/DiracDeterminantRef.cpp)

SUBDIRS(tests)
