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

namespace qmcplusplus
{

/** base class for Single-particle orbital sets
 *
 * SPOSet stands for S(ingle)P(article)O(rbital)Set which contains
 * a number of single-particle orbitals with capabilities of evaluating \f$ \phi_j({\bf r}_i)\f$
 */
class SPOSet : public QMCTraits
{
private:
  /// number of SPOs
  int OrbitalSetSize;
  /// name of the basis set
  std::string className;

public:
  /** return the size of the orbital set
   */
  inline int size() const
  {
    return OrbitalSetSize;
  }

  virtual void evaluate_v(const PosType &p) = 0;
  virtual void evaluate_vgl(const PosType &p) = 0;
  virtual void evaluate_vgh(const PosType &p) = 0;

};

}
#endif


