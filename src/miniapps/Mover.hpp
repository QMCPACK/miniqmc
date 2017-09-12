#ifndef QMCPLUSPLUS_WALKER_HPP
#define QMCPLUSPLUS_WALKER_HPP

#include <Configuration.h>
#include <Particle/ParticleSet.h>
#include <Utilities/RandomGenerator.h>
#include <miniapps/pseudo.hpp>
#include <spline2/MultiBsplineOffload.hpp>
#include <spline2/MultiBsplineRef.hpp>
#include <QMCWaveFunctions/einspline_spo.hpp>

namespace qmcplusplus
{
  struct Mover
  {
    using RealType = QMCTraits::RealType;
    using spo_type = einspline_spo<RealType, MultiBsplineOffload<RealType> >;
    using spo_ref_type = einspline_spo<RealType, MultiBsplineRef<RealType> >;

    RandomGenerator<RealType> *rng;
    ParticleSet               *els;
    spo_type                  *spo;
    spo_ref_type              *spo_ref;
    NonLocalPP<RealType>      *nlpp;
  };
}

#endif
