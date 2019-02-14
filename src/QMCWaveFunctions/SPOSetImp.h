////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2018 QMCPACK developers.
//
// File developed by:
// Peter Doak, doakpw@ornl.gov, Oak Ridge National Lab
//
// File created by:
// Peter Doak, doakpw@ornl.gov, Oak Ridge National Lab
////////////////////////////////////////////////////////////////////////////////

#ifndef QMCPLUSPLUS_SPO_SET_IMP_H
#define QMCPLUSPLUS_SPO_SET_IMP_H

#include "Devices.h"
#include "SPOSet.h"

/** @file
 * Here compiled SPOSet implmentations
 * are included.
 */

namespace qmcplusplus
{

template<Devices DT>
class SPOSetImp : public SPOSet
{
public:
  int size() const { return OrbitalSetSize; }
private:
  /// number of SPOs
  int OrbitalSetSize;
  /// name of the basis set
  std::string className;

public:
    /// operates on multiple walkers
  virtual void
  multi_evaluate_v(const std::vector<SPOSet*>& spo_list, const std::vector<PosType>& pos_list)
  {
    #pragma omp parallel for
    for (int iw = 0; iw < spo_list.size(); iw++)
      spo_list[iw]->evaluate_v(pos_list[iw]);
  }

  virtual void
  multi_evaluate_vgl(const std::vector<SPOSet*>& spo_list, const std::vector<PosType>& pos_list)
  {
    #pragma omp parallel for
    for (int iw = 0; iw < spo_list.size(); iw++)
      spo_list[iw]->evaluate_vgl(pos_list[iw]);
  }

  virtual void
  multi_evaluate_vgh(const std::vector<SPOSet*>& spo_list, const std::vector<PosType>& pos_list)
  {
    #pragma omp parallel for
    for (int iw = 0; iw < spo_list.size(); iw++)
      spo_list[iw]->evaluate_vgh(pos_list[iw]);
  }

};

}

#endif
