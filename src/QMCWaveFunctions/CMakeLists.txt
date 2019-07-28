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
            einspline_spo_omp.cpp
            WaveFunction.cpp SPOSet_builder.cpp
            DiracDeterminant.cpp DiracDeterminantRef.cpp)

SUBDIRS(tests)
