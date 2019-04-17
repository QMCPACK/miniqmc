#ifndef QMCPLUSPLUS_WAVEFUNCTION_BUILDER_H
#define QMCPLUSPLUS_WAVEFUNCTION_BUILDER_H
#include "Devices.h"
#include "Particle/ParticleSet.h"
#include "Utilities/RandomGenerator.h"
#include "Memory/DeviceBuffers.hpp"
#include "QMCWaveFunctions/WaveFunction.h"

namespace qmcplusplus
{
//class WaveFunction;

template<Devices DT>
class WaveFunctionBuilder
{
public:
  static void build(bool useRef,
                    WaveFunction& WF,
                    ParticleSet& ions,
                    ParticleSet& els,
                    const RandomGenerator<QMCTraits::RealType>& RNG,
                    DeviceBuffers<DT>& dev_bufs,
                    bool enableJ3);

private:
  static void devDetBuild(WaveFunction& WF,
                          const RandomGenerator<QMCTraits::RealType>& RNG,
                          ParticleSet& els,
                          DeviceBuffers<DT>& dev_bufs);
};

extern template class WaveFunctionBuilder<Devices::CPU>;

#ifdef QMC_USE_KOKKOS
extern template class WaveFunctionBuilder<Devices::KOKKOS>;
#endif

} // namespace qmcplusplus

#endif
