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

#include <QMCWaveFunctions/WaveFunctionKokkos.h>
#include <QMCWaveFunctions/WaveFunction.h>
#include <QMCWaveFunctions/Determinant.h>
#include <QMCWaveFunctions/Jastrow/BsplineFunctor.h>
#include <QMCWaveFunctions/Jastrow/OneBodyJastrow.h>
#include <QMCWaveFunctions/Jastrow/OneBodyJastrowKokkos.h>
#include <QMCWaveFunctions/Jastrow/TwoBodyJastrow.h>
#include <QMCWaveFunctions/Jastrow/TwoBodyJastrowKokkos.h>
#include <vector>

namespace qmcplusplus
{


WaveFunctionKokkos::WaveFunctionKokkos(const std::vector<WaveFunction*>& WF_list, int numKnots) {
  using J1OrbType = OneBodyJastrow<BsplineFunctor<ValueType>>;
  using J2OrbType = TwoBodyJastrow<BsplineFunctor<ValueType>>;

  const std::vector<WaveFunctionComponent*> upDetVector(extract_up_list(WF_list));
  const std::vector<WaveFunctionComponent*> downDetVector(extract_dn_list(WF_list));
  // note, this is assuming that j1 was added to the wavefunction before
  // j2.  Will work in our case, but would be better to do some testing
  // with something like dynamic cast somewhere.  
  const std::vector<WaveFunctionComponent*> oneBodyJastrowVector(extract_jas_list(WF_list, 0)); 
  const std::vector<WaveFunctionComponent*> twoBodyJastrowVector(extract_jas_list(WF_list, 1));

  // set the values of numElectrons and numIons
  numElectrons = static_cast<J1OrbType*>(oneBodyJastrowVector[0])->Nelec;
  numIons = static_cast<J1OrbType*>(oneBodyJastrowVector[0])->Nions;
  
  upDets = Kokkos::View<DiracDeterminantKokkos*>("upDets", upDetVector.size());
  downDets = Kokkos::View<DiracDeterminantKokkos*>("downDets", downDetVector.size());
  activeDDs = Kokkos::View<DiracDeterminantKokkos*>("activeDDs", upDetVector.size());
  oneBodyJastrows = Kokkos::View<objType*>("objs", oneBodyJastrowVector.size());
  twoBodyJastrows = Kokkos::View<tbjType*>("tjbs", twoBodyJastrowVector.size());
  isActive = Kokkos::View<int*>("wfc::isActive", upDetVector.size());
  activeMap = Kokkos::View<int*>("wfc::activeMap", upDetVector.size());

  // create mirrors
  auto upDetMirror = Kokkos::create_mirror_view(upDets);
  auto downDetMirror = Kokkos::create_mirror_view(downDets);
  auto objMirror = Kokkos::create_mirror_view(oneBodyJastrows);
  auto tbjMirror = Kokkos::create_mirror_view(twoBodyJastrows);
  activeMapMirror = Kokkos::create_mirror_view(activeMap);

  
  // note, here I am assuming that all input vectors have the same size!
  for (int i = 0; i < upDetVector.size(); i++) {
    upDetMirror(i) = static_cast<DiracDeterminant*>(upDetVector[i])->ddk;
    downDetMirror(i) = static_cast<DiracDeterminant*>(downDetVector[i])->ddk;
    objMirror(i) = static_cast<J1OrbType*>(oneBodyJastrowVector[i])->jasData;
    tbjMirror(i) = static_cast<J2OrbType*>(twoBodyJastrowVector[i])->jasData;
  }
  
  Kokkos::deep_copy(upDets, upDetMirror);
  Kokkos::deep_copy(downDets, downDetMirror);
  Kokkos::deep_copy(oneBodyJastrows, objMirror);
  Kokkos::deep_copy(twoBodyJastrows, tbjMirror);

  grad_view = Kokkos::View<ValueType**>("scratchGradView", upDetVector.size(), OHMMS_DIM);
  grad_view_mirror = Kokkos::create_mirror_view(grad_view);
  ratios_view = Kokkos::View<ValueType*>("scratchRatiosView", upDetVector.size());
  ratios_view_mirror = Kokkos::create_mirror_view(ratios_view);
  knots_ratios_view = Kokkos::View<ValueType**>("scratchKnotsRatiosView", upDetVector.size(), numKnots);
  knots_ratios_view_mirror = Kokkos::create_mirror_view(knots_ratios_view);
}

};
