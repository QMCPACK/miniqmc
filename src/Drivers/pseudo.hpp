////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by:
//
// File created by:
// Jeongnim Kim, jeongnim.kim@intel.com,
//    Intel Corp.
////////////////////////////////////////////////////////////////////////////////
// -*- C++ -*-
/** @file pseudo.hpp
 * @brief  extraction from NonLocalECPComponent
 */
#ifndef QMCPLUSPLUS_MINIAPPS_PSEUDO_H
#define QMCPLUSPLUS_MINIAPPS_PSEUDO_H

#include <Utilities/RandomGenerator.h>
#include <Particle/VirtualParticleSet.h>
#include <QMCWaveFunctions/WaveFunction.h>

namespace qmcplusplus
{
template<typename T>
struct NonLocalPP
{
  /** dimension */
  enum
  {
    D = 3
  };
  /** typedefs */
  using RealType   = T;
  using PosType    = TinyVector<T, D>;
  using TensorType = Tensor<T, D>;
  using ParticlePos_t = ParticleSet::ParticlePos_t;

  /** pseudo region, defined per specie in real simulation*/
  RealType Rmax;
  /** random number generator */
  RandomGenerator<RealType> myRNG;
  /** weight ot each point on the sphere */
  std::vector<RealType> weight_m;
  /** positions on a sphere */
  std::vector<PosType> sgridxyz_m;
  /** Virtual ParticleSet for each ionic specie*/
  std::vector<VirtualParticleSet> VPs;
  /** default constructor with knots=12 */
  NonLocalPP(const RandomGenerator<RealType>& rng) : myRNG(rng)
  {
    // use fixed seed
    myRNG.init(0, 1, 11);

    /** 12 knots */
    const int num_quadrature_points = 12;
    weight_m.resize(num_quadrature_points);
    sgridxyz_m.resize(num_quadrature_points);

    const RealType w = RealType(1.0 / num_quadrature_points);
    for (int i = 0; i < num_quadrature_points; ++i)
      weight_m[i] = w;

    // clang-format off
    sgridxyz_m[ 0] = PosType(           1.0,               0.0,               0.0 );
    sgridxyz_m[ 1] = PosType(          -1.0,               0.0,               0.0 );
    sgridxyz_m[ 2] = PosType(  0.4472135955,       0.894427191,               0.0 );
    sgridxyz_m[ 3] = PosType( -0.4472135955,      0.7236067977,      0.5257311121 );
    sgridxyz_m[ 4] = PosType(  0.4472135955,      0.2763932023,      0.8506508084 );
    sgridxyz_m[ 5] = PosType( -0.4472135955,     -0.2763932023,      0.8506508084 );
    sgridxyz_m[ 6] = PosType(  0.4472135955,     -0.7236067977,      0.5257311121 );
    sgridxyz_m[ 7] = PosType( -0.4472135955,      -0.894427191,               0.0 );
    sgridxyz_m[ 8] = PosType(  0.4472135955,     -0.7236067977,     -0.5257311121 );
    sgridxyz_m[ 9] = PosType( -0.4472135955,     -0.2763932023,     -0.8506508084 );
    sgridxyz_m[10] = PosType(  0.4472135955,      0.2763932023,     -0.8506508084 );
    sgridxyz_m[11] = PosType( -0.4472135955,      0.7236067977,     -0.5257311121 );
    // clang-format on
  }

  // create VPs
  void initialize_VPs(const ParticleSet& ions, const ParticleSet& elecs, const RealType Rmax_in)
  {
    if(VPs.size()) throw std::runtime_error("Can not create VPs again.\n");
    for (int i = 0; i < ions.groups(); ++i)
      VPs.emplace_back(elecs, size());
    Rmax = Rmax_in;
  }

  inline int size() const { return sgridxyz_m.size(); }

  template<typename PA>
  inline void randomize(PA& rrotsgrid)
  {
    // const RealType twopi(6.28318530718);
    // RealType phi(twopi*Random()),psi(twopi*Random()),cth(Random()-0.5),
    RealType phi(TWOPI * (myRNG())), psi(TWOPI * (myRNG())), cth((myRNG()) - 0.5);
    RealType sph(std::sin(phi)), cph(std::cos(phi)), sth(std::sqrt(1.0 - cth * cth)),
        sps(std::sin(psi)), cps(std::cos(psi));
    TensorType rmat(cph * cth * cps - sph * sps,
                    sph * cth * cps + cph * sps,
                    -sth * cps,
                    -cph * cth * sps - sph * cps,
                    -sph * cth * sps + cph * cps,
                    sth * sps,
                    cph * sth,
                    sph * sth,
                    cth);
    const int n = sgridxyz_m.size();
    for (int i = 0; i < n; ++i)
      rrotsgrid[i] = dot(rmat, sgridxyz_m[i]);
  }

  void evaluate(const ParticleSet& els, WaveFunction& wf)
  {
     ParticlePos_t rOnSphere(size());
     std::vector<QMCTraits::ValueType> ratios(size());
     randomize(rOnSphere); // pick random sphere
     const DistanceTableData* d_ie = els.DistTables[wf.get_ei_TableID()];
     const ParticleSet& ions(d_ie->origin());

     for (int jel = 0; jel < els.getTotalNum(); ++jel)
     {
       const auto& dist  = d_ie->Distances[jel];
       const auto& displ = d_ie->Displacements[jel];
       for (int iat = 0; iat < ions.getTotalNum(); ++iat)
       {
         if (dist[iat] < Rmax)
         {
           auto& VP = VPs[ions.GroupID[iat]];
           VP.makeMoves(jel, rOnSphere, true, iat);
           wf.evaluateRatios(VP, ratios);
         }
       }
     }

  }
};
} // namespace qmcplusplus
#endif
