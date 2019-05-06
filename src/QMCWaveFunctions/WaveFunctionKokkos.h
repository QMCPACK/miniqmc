////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by:
//
// File created by: Luke Shulenburger, lshulen@sandia.gov, Sandia National Laboratories
////////////////////////////////////////////////////////////////////////////////
// -*- C++ -*-


#ifndef QMCPLUSPLUS_WAVEFUNCTION_KOKKOS_H
#define QMCPLUSPLUS_WAVEFUNCTION_KOKKOS_H
#include <Kokkos_Core.hpp>
#include <Utilities/Configuration.h>
#include <QMCWaveFunctions/DeterminantKokkos.h>
#include <QMCWaveFunctions/Jastrow/OneBodyJastrowKokkos.h>
#include <QMCWaveFunctions/Jastrow/TwoBodyJastrowKokkos.h>
#include <vector>

namespace qmcplusplus
{
  /** Holds the collective views for the determinants and jastrows for a population
      of walkers */
class WaveFunction;

// no reason we couldn't add in some simple additional data like number of electrons
class WaveFunctionKokkos : public QMCTraits
{
public:
  using objType = OneBodyJastrowKokkos<RealType, OHMMS_DIM>;
  using tbjType = TwoBodyJastrowKokkos<RealType, ValueType, OHMMS_DIM>;

public:
  Kokkos::View<DiracDeterminantKokkos*> upDets;
  Kokkos::View<DiracDeterminantKokkos*> downDets;
  Kokkos::View<objType*> oneBodyJastrows;
  Kokkos::View<tbjType*> twoBodyJastrows;

  // add in some helpers to avoid allocating so much
  Kokkos::View<ValueType**> grad_view;
  Kokkos::View<ValueType**>::HostMirror grad_view_mirror;
  Kokkos::View<ValueType*> ratios_view;
  Kokkos::View<ValueType*>::HostMirror ratios_view_mirror;

  int numElectrons;
  int numIons;

 public:
  WaveFunctionKokkos(const std::vector<WaveFunction*>& WF_list);
};

}; // namespace


#endif 
