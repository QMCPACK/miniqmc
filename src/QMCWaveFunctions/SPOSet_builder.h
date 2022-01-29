// This file is distributed under the University of Illinois/NCSA Open Source
// License. See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//
// File created by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//////////////////////////////////////////////////////////////////////////////////////


#ifndef QMCPLUSPLUS_SINGLEPARTICLEORBITALSET_BUILDER_H
#define QMCPLUSPLUS_SINGLEPARTICLEORBITALSET_BUILDER_H

#include "QMCWaveFunctions/SPOSet.h"

namespace qmcplusplus
{
/// build the einspline SPOSet.
std::unique_ptr<SPOSet> build_SPOSet(bool useRef,
                     int nx,
                     int ny,
                     int nz,
                     int num_splines,
                     int nblocks,
                     const Tensor<OHMMS_PRECISION, 3>& lattice_b,
                     bool init_random = true);

/// build the einspline SPOSet as a view of the main one.
std::unique_ptr<SPOSet> build_SPOSet_view(bool useRef, const SPOSet& SPOSet_main, int team_size, int member_id);

} // namespace qmcplusplus
#endif
