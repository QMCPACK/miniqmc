//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by: Jeremy McMinnis, jmcminis@gmail.com, University of Illinois at Urbana-Champaign
//                    Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//                    Mark A. Berrill, berrillma@ornl.gov, Oak Ridge National Laboratory
//
// File created by: Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//////////////////////////////////////////////////////////////////////////////////////
    
    



/**@file ParticleSet.BC.cpp
 * @brief definition of functions controlling Boundary Conditions
 */
#include "Particle/ParticleSet.h"
#include "Particle/FastParticleOperators.h"
#include "Utilities/OhmmsInfo.h"
#include "Message/OpenMP.h"

namespace qmcplusplus
{

void ParticleSet::convert(const ParticlePos_t& pin, ParticlePos_t& pout)
{
  if(pin.getUnit() == pout.getUnit())
  {
    pout = pin;
    return;
  }
  if(pin.getUnit() == PosUnit::LatticeUnit)
    //convert to CartesianUnit
  {
    ConvertPosUnit<ParticlePos_t,Tensor_t,DIM,OHMMS_ORTHO>::apply(pin,Lattice.R,pout,0,pin.size());
  }
  else
    //convert to LatticeUnit
  {
    ConvertPosUnit<ParticlePos_t,Tensor_t,DIM,OHMMS_ORTHO>::apply(pin,Lattice.G,pout,0,pin.size());
  }
}

void ParticleSet::convert2Unit(const ParticlePos_t& pin, ParticlePos_t& pout)
{
  pout.setUnit(PosUnit::LatticeUnit);
  if(pin.getUnit() == PosUnit::LatticeUnit)
    pout = pin;
  else
    ConvertPosUnit<ParticlePos_t,Tensor_t,DIM,OHMMS_ORTHO>::apply(pin,Lattice.G,pout,0,pin.size());
}

void ParticleSet::convert2Cart(const ParticlePos_t& pin, ParticlePos_t& pout)
{
  pout.setUnit(PosUnit::CartesianUnit);
  if(pin.getUnit() == PosUnit::CartesianUnit)
    pout = pin;
  else
    ConvertPosUnit<ParticlePos_t,Tensor_t,DIM,OHMMS_ORTHO>::apply(pin,Lattice.R,pout,0,pin.size());
}

void ParticleSet::convert2Unit(ParticlePos_t& pinout)
{
  if(pinout.getUnit() == PosUnit::LatticeUnit)
    return;
  else
  {
    pinout.setUnit(PosUnit::LatticeUnit);
    ConvertPosUnit<ParticlePos_t,Tensor_t,DIM,OHMMS_ORTHO>::apply(pinout,Lattice.G,0,pinout.size());
  }
}

void ParticleSet::convert2Cart(ParticlePos_t& pinout)
{
  if(pinout.getUnit() == PosUnit::CartesianUnit)
    return;
  else
  {
    pinout.setUnit(PosUnit::CartesianUnit);
    ConvertPosUnit<ParticlePos_t,Tensor_t,DIM,OHMMS_ORTHO>::apply(pinout,Lattice.R,0,pinout.size());
  }
}

void ParticleSet::applyBC(const ParticlePos_t& pin, ParticlePos_t& pout)
{
  applyBC(pin,pout,0,pin.size());
}

void ParticleSet::applyBC(const ParticlePos_t& pin, ParticlePos_t& pout, int first, int last)
{
  const bool orthogonal = ParticleLayout_t::IsOrthogonal;
  int mode = pin.getUnit()*2+pout.getUnit();
  switch(mode)
  {
  case(0):
    ApplyBConds<ParticlePos_t,Tensor_t,DIM,orthogonal>::Cart2Cart(pin,Lattice.G,Lattice.R,pout,first,last);
    break;
  case(1):
    ApplyBConds<ParticlePos_t,Tensor_t,DIM,orthogonal>::Cart2Unit(pin,Lattice.G,pout,first,last);
    break;
  case(2):
    ApplyBConds<ParticlePos_t,Tensor_t,DIM,orthogonal>::Unit2Cart(pin,Lattice.R,pout,first,last);
    break;
  case(3):
    ApplyBConds<ParticlePos_t,Tensor_t,DIM,orthogonal>::Unit2Unit(pin,pout,first,last);
    break;
  }
}

void ParticleSet::applyBC(ParticlePos_t& pos)
{
  const bool orthogonal = ParticleLayout_t::IsOrthogonal;
  if(pos.getUnit()==PosUnit::LatticeUnit)
  {
    ApplyBConds<ParticlePos_t,Tensor_t,DIM,orthogonal>::Unit2Unit(pos,0,LocalNum);
  }
  else
  {
    ApplyBConds<ParticlePos_t,Tensor_t,DIM,orthogonal>::Cart2Cart(pos,Lattice.G,Lattice.R,0,LocalNum);
  }
}

void ParticleSet::applyMinimumImage(ParticlePos_t& pinout)
{
  if(Lattice.SuperCellEnum==SUPERCELL_OPEN)
    return;
  for(int i=0; i<pinout.size(); ++i)
    MinimumImageBConds<RealType,DIM>::apply(Lattice.R,Lattice.G,pinout[i]);
}

void ParticleSet::convert2UnitInBox(const ParticlePos_t& pin, ParticlePos_t& pout)
{
  pout.setUnit(PosUnit::LatticeUnit);
  convert2Unit(pin,pout); // convert to crystalline unit
  put2box(pout);
}

void ParticleSet::convert2CartInBox(const ParticlePos_t& pin, ParticlePos_t& pout)
{
  convert2UnitInBox(pin,pout); // convert to crystalline unit
  convert2Cart(pout);
}
}

