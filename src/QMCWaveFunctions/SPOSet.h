// This file is distributed under the University of Illinois/NCSA Open Source
// License. See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//
// File created by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//////////////////////////////////////////////////////////////////////////////////////


#ifndef QMCPLUSPLUS_SINGLEPARTICLEORBITALSET_H
#define QMCPLUSPLUS_SINGLEPARTICLEORBITALSET_H

#include <Utilities/Configuration.h>
#include <string>

#include "Devices.h"

namespace qmcplusplus
{
/** pure abstract base class for Single-particle orbital sets
 *
 * SPOSet stands for S(ingle)P(article)O(rbital)Set which contains
 * a number of single-particle orbitals with capabilities of evaluating \f$ \phi_j({\bf r}_i)\f$
 */
class SPOSet
{
public:
  //Type aliases
  using QMCT    = QMCTraits;
  using PosType = QMCT::PosType;

public:
  /// return the size of the orbital set
  virtual int size() const = 0;

  /// destructor
  virtual ~SPOSet() {}

  /// operates on a single walker
  /// evaluating SPOs
  virtual void evaluate_v(const PosType& p)   = 0;
  virtual void evaluate_vgl(const PosType& p) = 0;
  virtual void evaluate_vgh(const PosType& p) = 0;

  /// operates on multiple walkers
  virtual void multi_evaluate_v(const std::vector<SPOSet*>& spo_list, const std::vector<PosType>& pos_list) = 0;

  virtual void multi_evaluate_vgl(const std::vector<SPOSet*>& spo_list, const std::vector<PosType>& pos_list) = 0;

  virtual void multi_evaluate_vgh(const std::vector<SPOSet*>& spo_list, const std::vector<PosType>& pos_list) = 0;
};

} // namespace qmcplusplus
#endif
