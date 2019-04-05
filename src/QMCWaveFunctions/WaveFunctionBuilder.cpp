#include "QMCWaveFunctions/WaveFunctionBuilder.h"
#include "QMCWaveFunctions/WaveFunction.h"
#include "QMCWaveFunctions/Jastrow/BsplineFunctor.h"
#include "QMCWaveFunctions/Jastrow/BsplineFunctorRef.h"
#include "QMCWaveFunctions/Jastrow/OneBodyJastrow.h"
#include "QMCWaveFunctions/Jastrow/OneBodyJastrowRef.h"
#include "QMCWaveFunctions/Jastrow/TwoBodyJastrow.h"
#include "QMCWaveFunctions/Jastrow/TwoBodyJastrowRef.h"
#include "QMCWaveFunctions/Jastrow/ThreeBodyJastrow.h"
#include "QMCWaveFunctions/Jastrow/ThreeBodyJastrowRef.h"
#include "QMCWaveFunctions/Jastrow/PolynomialFunctor3D.h"
#include "QMCWaveFunctions/Determinant.h"
#include "QMCWaveFunctions/DeterminantDevice.h"
#include "QMCWaveFunctions/DeterminantDeviceImp.h"
#include "QMCWaveFunctions/DeterminantRef.h"
#include <Input/Input.hpp>

// #ifdef QMC_USE_CUDA
// #include "QMCWaveFunctions/WaveFunctionBuilderCUDA.hpp"
// #endif


namespace qmcplusplus
{
template<Devices DT>
void WaveFunctionBuilder<DT>::devDetBuild(WaveFunction& WF, const RandomGenerator<QMCTraits::RealType>& RNG,
		                                ParticleSet& els,DeviceBuffers<DT>& dev_bufs)
{
    using DetType   = DiracDeterminant<DeterminantDeviceImp<DT>>;

    // determinant component
    WF.Det_up = new DetType(WF.nelup, RNG, 0);
    WF.Det_dn = new DetType(els.getTotalNum() - WF.nelup, RNG, WF.nelup);

}
    
template<Devices DT>
void WaveFunctionBuilder<DT>::build(bool useRef,
                                    WaveFunction& WF,
                                    ParticleSet& ions,
                                    ParticleSet& els,
                                    const RandomGenerator<QMCTraits::RealType>& RNG,
				    DeviceBuffers<DT>* dev_bufs,
                                    bool enableJ3)
{
  using valT = WaveFunction::valT;
  using posT = WaveFunction::posT;

  if (WF.Is_built)
  {
    app_log() << "The wavefunction was built before!" << std::endl;
    return;
  }

  const int nelup = els.getTotalNum() / 2;

  if (useRef)
  {
    using J1OrbType = miniqmcreference::OneBodyJastrowRef<BsplineFunctorRef<valT>>;
    using J2OrbType = miniqmcreference::TwoBodyJastrowRef<BsplineFunctorRef<valT>>;
    using J3OrbType = miniqmcreference::ThreeBodyJastrowRef<PolynomialFunctor3D>;
    using DetType   = miniqmcreference::DiracDeterminantRef;

    ions.RSoA = ions.R;
    els.RSoA  = els.R;

    // distance tables
    els.addTable(els, DT_SOA);
    WF.ei_TableID = els.addTable(ions, DT_SOA);

 	WF.nelup  = nelup;

WF.Det_up = new DetType(nelup, RNG, 0);
    WF.Det_dn = new DetType(els.getTotalNum() - nelup, RNG, nelup);
    
    // J1 component
    J1OrbType* J1 = new J1OrbType(ions, els);
    buildJ1(*J1, els.Lattice.WignerSeitzRadius);
    WF.Jastrows.push_back(J1);

    // J2 component
    J2OrbType* J2 = new J2OrbType(els);
    buildJ2(*J2, els.Lattice.WignerSeitzRadius);
    WF.Jastrows.push_back(J2);

    // J3 component
    if (enableJ3)
    {
      J3OrbType* J3 = new J3OrbType(ions, els);
      buildJeeI(*J3, els.Lattice.WignerSeitzRadius);
      WF.Jastrows.push_back(J3);
    }
  }
  else
  {
    using J1OrbType = OneBodyJastrow<DT, BsplineFunctor<DT, valT>>;
    using J2OrbType = TwoBodyJastrow<DT, BsplineFunctor<DT, valT>>;
    using J3OrbType = ThreeBodyJastrow<DT, PolynomialFunctor3D>;
    ions.RSoA       = ions.R;
    els.RSoA        = els.R;

    // distance tables
    els.addTable(els, DT_SOA);
    WF.ei_TableID = els.addTable(ions, DT_SOA);

    // determinant component
    WF.nelup  = nelup;
    devDetBuild(WF, RNG, els,  *dev_bufs);

    // J1 component
    J1OrbType* J1 = new J1OrbType(ions, els);
    buildJ1(*J1, els.Lattice.WignerSeitzRadius);
    WF.Jastrows.push_back(J1);

    // J2 component
    J2OrbType* J2 = new J2OrbType(els);
    buildJ2(*J2, els.Lattice.WignerSeitzRadius);
    WF.Jastrows.push_back(J2);

    // J3 component
    if (enableJ3)
    {
      J3OrbType* J3 = new J3OrbType(ions, els);
      buildJeeI(*J3, els.Lattice.WignerSeitzRadius);
      WF.Jastrows.push_back(J3);
    }
  }

  WF.setupTimers();
  WF.Is_built = true;
}

template class WaveFunctionBuilder<Devices::CPU>;
    // template void WaveFunctionBuilder<Devices::CPU>::devDetBuild(WaveFunction& WF, const RandomGenerator<QMCTraits::RealType>& RNG,
    // 								 ParticleSet& els,DeviceBuffers<Devices::CPU>& dev_bufs);
#ifdef QMC_USE_KOKKOS
template class WaveFunctionBuilder<Devices::KOKKOS>;
#endif
#ifdef QMC_USE_CUDA
template class WaveFunctionBuilder<Devices::CUDA>;
#endif

} // namespace qmcplusplus
