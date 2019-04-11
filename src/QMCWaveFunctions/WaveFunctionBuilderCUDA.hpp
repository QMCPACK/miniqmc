#ifndef QMCPLUSPLUS_WAVEFUNCTION_BUILDER_CUDA_H
#define QMCPLUSPLUS_WAVEFUNCTION_BUILDER_CUDA_H
#include "Devices.h"
#include "QMCWaveFunctions/WaveFunction.h"
#include "QMCWaveFunctions/Determinant.h"
#include "QMCWaveFunctions/DeterminantDeviceImp.h"

#include "QMCWaveFunctions/WaveFunctionBuilder.h"
namespace qmcplusplus
{
    template <>
    inline void WaveFunctionBuilder<Devices::CUDA>::devDetBuild(WaveFunction& WF,const RandomGenerator<QMCTraits::RealType>& RNG,                                     ParticleSet& els,DeviceBuffers<Devices::CUDA>& dev_bufs)
    {
	if(els.getTotalNum() < 2048)
	{
	    using DetType   = DiracDeterminant<DeterminantDeviceImp<Devices::CPU>>;

	    // determinant component
	    WF.Det_up = new DetType(WF.nelup, RNG, 0);
	    WF.Det_dn = new DetType(els.getTotalNum() - WF.nelup, RNG, WF.nelup);
	}
	else
	{
	    using DetType   = DiracDeterminant<DeterminantDeviceImp<Devices::CUDA>>;

	    // determinant component
	    WF.Det_up = new DetType(WF.nelup, RNG, dev_bufs, 0);
	    WF.Det_dn = new DetType(els.getTotalNum() - WF.nelup, RNG, dev_bufs, WF.nelup);
	}
    }

}

#endif
