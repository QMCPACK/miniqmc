// This file is distributed under the University of Illinois/NCSA Open Source
// License. See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//
// File created by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//////////////////////////////////////////////////////////////////////////////////////


#include "QMCWaveFunctions/SPOSet_builder.h"
#include <Utilities/RandomGenerator.h>
#include "QMCWaveFunctions/einspline_spo_omp.hpp"
#include "QMCWaveFunctions/einspline_spo_ref.hpp"

namespace qmcplusplus
{
std::unique_ptr<SPOSet> build_SPOSet(bool useRef,
                                     int nx,
                                     int ny,
                                     int nz,
                                     int num_splines,
                                     int nblocks,
                                     const Tensor<OHMMS_PRECISION, 3>& lattice_b,
                                     bool init_random)
{
  if (useRef)
  {
    auto spo_main = std::make_unique<miniqmcreference::einspline_spo_ref<OHMMS_PRECISION>>();
    spo_main->set(nx, ny, nz, num_splines, nblocks);
    spo_main->Lattice.set(lattice_b);
    return spo_main;
  }
  else
  {
    auto spo_main = std::make_unique<einspline_spo_omp<OHMMS_PRECISION>>();
    spo_main->set(nx, ny, nz, num_splines, nblocks);
    spo_main->Lattice.set(lattice_b);
    return spo_main;
  }
}

std::unique_ptr<SPOSet> build_SPOSet_view(bool useRef, const SPOSet& SPOSet_main, int team_size, int member_id)
{
  if (useRef)
  {
    auto& temp_ptr = dynamic_cast<const miniqmcreference::einspline_spo_ref<OHMMS_PRECISION>&>(SPOSet_main);
    auto spo_view =
        std::make_unique<miniqmcreference::einspline_spo_ref<OHMMS_PRECISION>>(temp_ptr, team_size, member_id);
    return spo_view;
  }
  else
  {
    auto& temp_ptr = dynamic_cast<const einspline_spo_omp<OHMMS_PRECISION>&>(SPOSet_main);
    auto spo_view  = std::make_unique<einspline_spo_omp<OHMMS_PRECISION>>(temp_ptr, team_size, member_id);
    return spo_view;
  }
}

} // namespace qmcplusplus
