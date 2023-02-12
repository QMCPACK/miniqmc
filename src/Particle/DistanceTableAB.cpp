////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by:
// Ken Esler, kpesler@gmail.com,
//    University of Illinois at Urbana-Champaign
// Jeremy McMinnis, jmcminis@gmail.com,
//    University of Illinois at Urbana-Champaign
// Jeongnim Kim, jeongnim.kim@gmail.com,
//    University of Illinois at Urbana-Champaign
// Jaron T. Krogel, krogeljt@ornl.gov,
//    Oak Ridge National Laboratory
// Mark A. Berrill, berrillma@ornl.gov,
//    Oak Ridge National Laboratory
//
// File created by:
// Jeongnim Kim, jeongnim.kim@gmail.com,
//    University of Illinois at Urbana-Champaign
////////////////////////////////////////////////////////////////////////////////

#include "Utilities/OpenMP.h"
#include "Particle/DistanceTable.h"
#include "Particle/DistanceTableData.h"
#include "Particle/Lattice/ParticleBConds.h"
#include "CPU/SIMD/algorithm.hpp"
#include "Particle/DistanceTableBA.h"

namespace qmcplusplus
{
/** Adding AsymmetricDTD to the list, e.g., el-el distance table
 *\param s source/target particle set
 *\return index of the distance table with the name
 */
DistanceTableData* createDistanceTable(const ParticleSet& s, ParticleSet& t)
{
  typedef OHMMS_PRECISION RealType;
  enum
  {
    DIM = OHMMS_DIM
  };
  DistanceTableData* dt = 0;
  int sc                = t.Lattice.SuperCellEnum;
  std::ostringstream o;
  o << "  Distance table for AB: source = " << s.getName() << " target = " << t.getName() << "\n";
  if (sc == SUPERCELL_BULK)
  {
    o << "  Using SoaDistanceTableBA<T,D,PPPG> of SoA layout " << PPPG << std::endl;
    dt = new DistanceTableBA<RealType, DIM, PPPG + SOA_OFFSET>(s, t);
    o << "    Setting Rmax = " << s.Lattice.SimulationCellRadius;
  }
  else
  {
    APP_ABORT("DistanceTableData::createDistanceTable Slab/Wire/Open boundary "
              "conditions are disabled in miniQMC!\n");
  }

  o << " using Cartesian coordinates";
  if (omp_get_thread_num() == 0)
  {
    app_log() << o.str() << std::endl;
    app_log().flush();
  }
  return dt;
}

} // namespace qmcplusplus
/***************************************************************************
 * $RCSfile$   $Author$
 * $Revision$   $Date$
 * $Id$
 ***************************************************************************/
