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
// -*- C++ -*-

/**
 * @file
 * @brief Shared definitions for EinsplineSPODevices
 */

#ifndef QMCPLUSPLUS_EINSPLINE_SPO_PARAMS_H
#define QMCPLUSPLUS_EINSPLINE_SPO_PARAMS_H

namespace qmcplusplus
{
template<typename T>
struct EinsplineSPOParams
{
  /// number of blocks
  int nBlocks;
  /// first logical block index
  int firstBlock;
  /// last logical block index
  int lastBlock;





  /// number of splines
  int nSplines;
  /// number of splines per block
  int nSplinesPerBlock;
  int nSplinesSerialThreshold_V;
  int nSplinesSerialThreshold_VGH;
  

  /// if true, responsible for cleaning up einsplines
  bool Owner;
  /// if true, responsible for cleaning up host side einsplines
  bool host_owner;
  /// if true, is copy.  For reference counting & clean up in Kokkos.
  bool is_copy;

  CrystalLattice<T, 3> lattice;
};
  
}

#endif
