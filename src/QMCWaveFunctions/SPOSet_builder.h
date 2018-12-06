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

#include "QMCWaveFunctions/SPOSetImp.h"
#include "QMCWaveFunctions/EinsplineSPO.hpp"
#include "QMCWaveFunctions/einspline_spo_ref.hpp"

namespace qmcplusplus
{
/// build the einspline SPOSet.
template<Devices DT>
class SPOSetBuilder
{
public:
  static SPOSet* build(bool useRef,
			  int nx,
			  int ny,
			  int nz,
			  int num_splines,
			  int nblocks,
			  const Tensor<OHMMS_PRECISION, 3>& lattice_b,
			  bool init_random = true)
    {
  if (useRef)
  {
    miniqmcreference::EinsplineSPO_ref<OHMMS_PRECISION>* spo_main = new miniqmcreference::EinsplineSPO_ref<OHMMS_PRECISION>;
    spo_main->set(nx, ny, nz, num_splines, nblocks);
    spo_main->Lattice.set(lattice_b);
    return dynamic_cast<SPOSet*>(spo_main);
  }
  else
  {
    EinsplineSPO<DT, OHMMS_PRECISION>* spo_main = new EinsplineSPO<DT, OHMMS_PRECISION>;
    spo_main->set(nx, ny, nz, num_splines, nblocks);
    spo_main->setLattice(lattice_b);
    return dynamic_cast<SPOSet*>(spo_main);
  }
}

/// build the einspline SPOSet as a view of the main one.
  static SPOSet* buildView(bool useRef, const SPOSet* SPOSet_main, int team_size, int member_id)
    {
  if (useRef)
  {
    auto* temp_ptr =
      dynamic_cast<const miniqmcreference::EinsplineSPO_ref<OHMMS_PRECISION>*>(SPOSet_main);
    auto* spo_view =
      new miniqmcreference::EinsplineSPO_ref<OHMMS_PRECISION>(*temp_ptr, team_size, member_id);
    return dynamic_cast<SPOSet*>(spo_view);
  }
  else
  {
    auto* temp_ptr = dynamic_cast<const EinsplineSPO<DT, OHMMS_PRECISION>*>(SPOSet_main);
    auto* spo_view = new EinsplineSPO<DT, OHMMS_PRECISION>(*temp_ptr, team_size, member_id);
    return dynamic_cast<SPOSet*>(spo_view);
  }
}

};


} // namespace qmcplusplus
#endif
