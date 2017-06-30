//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by: Ken Esler, kpesler@gmail.com, University of Illinois at Urbana-Champaign
//                    Jeremy McMinnis, jmcminis@gmail.com, University of Illinois at Urbana-Champaign
//                    Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//                    Mark A. Berrill, berrillma@ornl.gov, Oak Ridge National Laboratory
//
// File created by: Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//////////////////////////////////////////////////////////////////////////////////////
    
    



#include "Lattice/Uniform3DGridLayout.h"
#include "Utilities/OhmmsInfo.h"
#include "Message/OpenMP.h"
#include <limits>

using namespace qmcplusplus;

///Set LR_rc = radius of smallest sphere inside box and kc=dim/rc
void Uniform3DGridLayout::SetLRCutoffs()
{
  //Compute rc as the real-space cutoff of 1/2 the unit-cell.
  //Radius of maximum shere that fits in a...
  TinyVector<value_type,3> b,c,d,x; //Unit vector of each surface will be in here
  //Compute the coordinate of the box center
  c = 0.5*(a(0)+a(1)+a(2));
  LR_rc = 1.e+6;
  Tensor<int,3> Cyclic(0,1,2, 1,2,0, 2,0,1);
  for(int i=0; i<3; i++)
  {
    TinyVector<value_type,3> v1 = a(Cyclic(i,1));
    TinyVector<value_type,3> v2 = a(Cyclic(i,2));
    value_type beta1 = (dot(v2,v2)*dot(c,v1) - dot (v1,v2)*dot(c,v2))/
                       (dot(v1,v1)*dot(v2,v2) - dot(v1,v2)*dot(v1,v2));
    value_type beta2 = (dot(v1,v1)*dot(c,v2) - dot (v1,v2)*dot(c,v1))/
                       (dot(v1,v1)*dot(v2,v2) - dot(v1,v2)*dot(v1,v2));
    TinyVector<value_type,3> p = beta1*v1 + beta2 * v2;
    value_type dist = sqrt (dot(p-c,p-c));
    LR_rc = std::min(LR_rc,dist);
//       b = cross(a(Cyclic(i,1)),a(Cyclic(i,2)));
//       value_type binv=1.0/std::sqrt(dot(b,b));
//       b *= binv;
//       ////Unit vector normal to surface i. Cyclic permutations of i.
//       //b = cross(a(i-2<0?i-2+3:i-2),a(i-1<0?i-1+3:i-1));
//       //b = b/std::sqrt(b[0]*b[0] + b[1]*b[1] + b[2]*b[2]);
//       //Now find multiple of 'b' that moves to centre of box
//       //d = 0.5*(a(i-2<0?i-2+3:i-2)+a(i-1<0?i-1+3:i-1))-c;
//       d = 0.5*(a(Cyclic(i,1))+a(Cyclic(i,2)))-c;
//       x[i]=1.e+6;
//       for(int l=0;l<3;l++){
// 	if(std::abs(d[l]*b[l]) < 1.e-6) continue; //Don't treat 0 elements.
// 	d[l] = d[l]/b[l];
// 	x[i] = std::min(x[i],std::abs(d[l]));
//       }
//       //Real-space cutoff is minimal x[i] => sphere fits entirely inside cell.
//       LR_rc = std::min(LR_rc,x[i]);
  }
  //Set KC for structure-factor and LRbreakups.
  LR_kc = LR_dim_cutoff/LR_rc;
  LOGMSG("\tLong-range breakup parameters:");
  LOGMSG("\trc*kc = " << LR_dim_cutoff << "; rc = " << LR_rc << "; kc = " << LR_kc << "\n");
  LOGMSG("\tWignerSeitzRadius = " << WignerSeitzRadius << "\n");
  LOGMSG("\tSimulationCellRadius = " << SimulationCellRadius << "\n");
}

void Uniform3DGridLayout::print(std::ostream& os) const
{
  os << "<unitcell>" << std::endl;
  Base_t::print(os);
  os << "<note>" << std::endl;
  os << "\tLong-range breakup parameters:" << std::endl;
  os << "\trc*kc = " << LR_dim_cutoff << "; rc = " << LR_rc << "; kc = " << LR_kc << "\n" << std::endl;
  os << "</note>" << std::endl;
  os << "</unitcell>" << std::endl;
  ////printGrid(os);
  //for(int ig=0; ig<c_offset.size()-1; ig++) {
  //  os << ig << " has neighboring cell "
  //     << c_max[ig]-c_offset[ig] << " "
  //     << c_offset[ig+1]-c_offset[ig]<< std::endl;
  //  //for(int ii=c_offset[ig]; ii<c_max[ig]; ii++) {
  //  //  os << c_id[ii] << " " << c_bc[ii] << std::endl;
  //  //}
  //}
}

