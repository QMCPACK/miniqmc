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

#include "Particle/DistanceTable.h"
#include "Particle/DistanceTableData.h"
#include "Particle/Lattice/ParticleBConds.h"
#include "Particle/DistanceTableAA.h"

namespace qmcplusplus
{

/** Adding SymmetricDTD to the list, e.g., el-el distance table
 *\param s source/target particle set
 *\return index of the distance table with the name
 */
DistanceTableData *createDistanceTable(ParticleSet &s, int dt_type)
{
  typedef OHMMS_PRECISION RealType;
  enum
  {
    DIM = OHMMS_DIM
  };
  int sc                = s.Lattice.SuperCellEnum;
  DistanceTableData *dt = 0;
  std::ostringstream o;
  bool useSoA = (dt_type == DT_SOA || dt_type == DT_SOA_PREFERRED);
  o << "  Distance table for AA: source/target = " << s.getName()
    << " useSoA =" << useSoA << "\n";
  if (sc == SUPERCELL_BULK)
  {
    o << "  Using SoaDistanceTableAA<T,D,PPPG> of SoA layout " << PPPG
      << std::endl;
    dt = new DistanceTableAA<RealType, DIM, PPPG + SOA_OFFSET>(s);
    o << "\n    Setting Rmax = " << s.Lattice.SimulationCellRadius;
  }
  else
  {
    APP_ABORT("DistanceTableData::createDistanceTable Slab/Wire/Open boundary "
              "conditions are disabled in miniQMC!\n");
  }

  // set dt properties
  dt->CellType = sc;
  dt->DTType   = DT_SOA;
  std::ostringstream p;
  p << s.getName() << "_" << s.getName();
  dt->Name = p.str(); // assign the table name

  o << " using Cartesian coordinates";
  if (qmc_get_thread_num() == 0)
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
