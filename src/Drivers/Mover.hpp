////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by:
// Ye Luo, yeluo@anl.gov,
//    Argonne National Laboratory
//
// File created by:
// Ye Luo, yeluo@anl.gov,
//    Argonne National Laboratory
////////////////////////////////////////////////////////////////////////////////


#ifndef QMCPLUSPLUS_MOVER_HPP
#define QMCPLUSPLUS_MOVER_HPP

#include <Utilities/Configuration.h>
#include <Utilities/RandomGenerator.h>
#include <Particle/ParticleSet.h>
#include <QMCWaveFunctions/einspline_spo.hpp>
#include <Numerics/Spline2/MultiBspline.hpp>
#include <QMCWaveFunctions/WaveFunction.h>
#include <Input/pseudo.hpp>

namespace qmcplusplus
{
  struct Mover
  {
    using RealType = QMCTraits::RealType;
    using spo_type = einspline_spo<RealType, MultiBspline<RealType> >;

    RandomGenerator<RealType> rng;
    ParticleSet               ions;
    ParticleSet               els;
    spo_type                  spo;
    WaveFunction              wavefunction;
    NonLocalPP<RealType>      nlpp;

    Mover(const spo_type &spo_main,
          const int team_size,
          const int member_id,
          const uint32_t myPrime,
          const ParticleSet &ions);

  };

  const int build_ions(ParticleSet &ions,
                       const Tensor<int, 3> &tmat,
                       Tensor<QMCTraits::RealType, 3> &lattice);

  const int build_els(ParticleSet &els,
                      const ParticleSet &ions,
                      RandomGenerator<QMCTraits::RealType> &rng);

  template <class T>
  const std::vector<T *> filtered_list(const std::vector<T *> &input_list, const std::vector<bool> &chosen)
  {
    std::vector<T *> final_list;
    for(int iw=0; iw<input_list.size(); iw++)
      if(chosen[iw]) final_list.push_back(input_list[iw]);
    return final_list;
  }

  const std::vector<ParticleSet *> extract_els_list(const std::vector<Mover *> &mover_list);

  const std::vector<WaveFunction *> extract_wf_list(const std::vector<Mover *> &mover_list);
}

#endif
