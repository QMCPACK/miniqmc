////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2019 QMCPACK developers.
//
// File developed by:
// Peter Doak, doakpw@ornl.gov, Oak Ridge National Lab
// Jeongnim Kim, jeongnim.kim@intel.com,
//    Intel Corp.
// Amrita Mathuriya, amrita.mathuriya@intel.com,
//    Intel Corp.
//
// File created by:
// Jeongnim Kim, jeongnim.kim@intel.com,
//    Intel Corp.
////////////////////////////////////////////////////////////////////////////////
// -*- C++ -*-
// clang-format off
/** @file
    @brief Miniapp to capture the computation in particle moves.

 @mainpage MiniQMC: miniapp for QMCPACK kernels

 Implemented kernels
   - \subpage JastrowFactors "Jastrow Factors" includes one-body, two-body and three-body Jastrow
     factors.
   - \subpage SPO "Single Particle Orbitals" (SPO) based on splines
   - \subpage InverseUpdate "Inverse matrix update" for determinant
   - \subpage ParticleHandling "Particle distances" and boundary conditions



  The \ref src/Drivers/miniqmc.cpp "miniqmc" application models a complete block of QMC steps with particle moves and evaluation of the wavefunction.
  The <a href="https://github.com/QMCPACK/miniqmc/wiki#miniqmc-computational-overview">wiki</a> gives an outline of the computation.

  The \ref src/Drivers/check_wfc.cpp "check_wfc", \ref src/Drivers/check_spo.cpp "check_spo",
  and \ref src/Drivers/check_determinant.cpp "check_determinant" drivers check correctness by comparing against
  reference implementations of the Jastrow, SPO, and determinant inverse (respectively).
  The code for the reference implementation uses a `Ref` suffix and is contained in the \ref miniqmcreference namespace.


 */

 /*!
 \page JastrowFactors Jastrow Factors

  The Jastrow factor accounts for the correlation of electron-ion pairs
  (one-body Jastrow), two-electron pairs (two-body Jastrow) and
  two-electron-one-ion trios (three-body/eeI Jastrow).

  The Jastrow factor is composed from two types of classes - the first is for
  the types of particles involved (one/two/three body), and the second is the
  functional form for the radial part.  The classes for the first part are
  qmcplusplus::OneBodyJastrow, qmcplusplus::TwoBodyJastrow and
  qmcplusplus::ThreeBodyJastrow.  The second part uses 1D B-splines, defined
  in qmcplusplus::BsplineFunctor, for one and two body Jastrow and polynomials,
  defined in qmcplusplus::PolynomialFunctor3D, for three body Jastrow.

  This miniapp only contains the B-spline and polynomial functional form, since
  it is the most widely used.  The QMCPACK distribution contains other
  functional forms.
 */

 /*!
 \page SPO Single Particle Orbitals

  The Single Particle Orbitals (SPO) depend only on individual electron coordinates and are represented by a 3D spline.
  The 3D spline code is located in \ref src/Numerics/Spline2, with the evaluation code in the \ref qmcplusplus::MultiBspline "MultiBspline" class.
  The connection from the wavefunction to the spline functions is located in the \ref qmcplusplus::EinsplineSPO "EinsplineSPO" class.

  The core evaluation routine evaluates the value, the gradient, and the Laplacian at a given electron coordinate.

  The size of the coefficient data set can be large - on the order of gigabytes.

 */

 /*!
 \page InverseUpdate Inverse Matrix Update

  The inverse matrix and updating is handled by \ref qmcplusplus::DiracDeterminant "DiracDeterminant".
  The initial creation and inversion of the matrix occurs in qmcplusplus::DiracDeterminant::recompute, which is called on the first
  call to qmcplusplus::DiracDeterminant::evaluateLog.   The rows are updated after accepted Monte Carlo moves
  via updateRow (in \ref src/QMCWaveFunctions/Determinant.h), which is called from qmcplusplus::DiracDeterminant::acceptMove.

  The implementation for updateRow is
  \snippet QMCWaveFunctions/Determinant.h UpdateRow
 */

 /*!
 \page ParticleHandling Particle positions, distances, and boundary conditions

  The qmcpluplus:ParticleSet class holds particle positions, lattice information, and distance tables.
  The positions are stored in the \ref qmcplusplus::ParticleSet#R member, and a copy is kept in a Structure-of-Arrays (SoA) layout in
  \ref qmcplusplus::ParticleSet#RSoA.

  Distances, using the minimum image conventions with periodic boundaries, are computed in \ref src/Particle/Lattice/ParticleBConds.h.

  Distances are stored in distance tables, where qmcplusplus::DistanceTableData is the base class for the storage.  There are two types
  of distance tables.  One is for similar particles (qmcplusplus::DistanceTableAA), such as electron-electron distances.  The other
  is for dissimilar particles (qmcplusplus::DistanceTableBA), such as electron-ion distances.
 */

// clang-format on

#include "Drivers/MiniqmcDriver.h"
#include "Drivers/MiniqmcOptions.hpp"
//using namespace qmcplusplus;

using namespace std;
using namespace qmcplusplus;
int main(int argc, char** argv)
{
  //int my_int;
  //std::cout << my_int;
  try
  {
    MiniqmcOptions mq_opt = readOptions(argc, argv);
    if (mq_opt.valid)
    {
      MiniqmcDriver driver(mq_opt);
      driver.initialize(argc, argv);
      driver.run();
    }
  }
  catch (int e)
  {
    return e;
  }
  return 0;
}
