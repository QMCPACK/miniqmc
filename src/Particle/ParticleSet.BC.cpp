////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by:
// Jeremy McMinnis, jmcminis@gmail.com,
//    University of Illinois at Urbana-Champaign
// Jeongnim Kim, jeongnim.kim@gmail.com,
//    University of Illinois at Urbana-Champaign
// Mark A. Berrill, berrillma@ornl.gov,
//    Oak Ridge National Laboratory
//
// File created by:
// Jeongnim Kim, jeongnim.kim@gmail.com,
//    University of Illinois at Urbana-Champaign
////////////////////////////////////////////////////////////////////////////////

/**@file ParticleSet.BC.cpp
 * @brief definition of functions controlling Boundary Conditions
 */
#include "Particle/ParticleSet.h"
#include "Particle/FastParticleOperators.h"

namespace qmcplusplus
{
void ParticleSet::convert2Unit(ParticlePos_t& pinout)
{
  if (pinout.getUnit() == PosUnit::LatticeUnit)
    return;
  else
  {
    pinout.setUnit(PosUnit::LatticeUnit);
    ConvertPosUnit<ParticlePos_t, Tensor_t, DIM, OHMMS_ORTHO>::apply(pinout,
                                                                     Lattice.G,
                                                                     0,
                                                                     pinout.size());
  }
}

void ParticleSet::convert2Cart(ParticlePos_t& pinout)
{
  if (pinout.getUnit() == PosUnit::CartesianUnit)
    return;
  else
  {
    pinout.setUnit(PosUnit::CartesianUnit);
    ConvertPosUnit<ParticlePos_t, Tensor_t, DIM, OHMMS_ORTHO>::apply(pinout,
                                                                     Lattice.R,
                                                                     0,
                                                                     pinout.size());
  }
}
} // namespace qmcplusplus
