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
#ifndef QMCPLUSPLUS_CHECK_SPO_STEPS_KOKKOS_HPP
#define QMCPLUSPLUS_CHECK_SPO_STEPS_KOKKOS_HPP

#include "Drivers/check_spo.h"
#include "Drivers/CheckSPOSteps.hpp"
#include "Utilities/Configuration.h"

namespace qmcplusplus
{

template<>
inline void CheckSPOSteps<Devices::KOKKOS>::finalize()
{
  Kokkos::finalize();
}

template<>
inline void CheckSPOSteps<Devices::KOKKOS>::initialize(int argc, char** argv)
{
  std::cout << "CheckSPOSteps<DDT::KOKKOS>::initialize" << '\n';
  Kokkos::initialize(argc, argv);
}

    /** Kokkos functor for custom reduction
 */
template<typename T>
class SPOReduction
{
public:
  using QMCT = QMCTraits;
  using SPODevImp  = EinsplineSPO<Devices::KOKKOS, OHMMS_PRECISION>;
  using SPORef     = miniqmcreference::EinsplineSPO_ref<OHMMS_PRECISION>;
  using value_type = CheckSPOData<T>;
  using size_type = int;
  SPOReduction(const int team_size,
                const ParticleSet ions,
                const SPODevImp spo_main,
                const SPORef spo_ref_main,
                const int nsteps,
	       const QMCT::RealType Rmax)
      : team_size_(team_size),
        ions_(ions),
        spo_main_(spo_main),
        spo_ref_main_(spo_ref_main),
        nsteps_(nsteps),
        Rmax_(Rmax)
  {
#ifdef KOKKOS_ENABLE_OPENMP
    ncrews_ = 	Kokkos::OpenMP::thread_pool_size();
#else
    ncrews_ = 1;
#endif
    
    crewsize_    = 1;
  }

  KOKKOS_INLINE_FUNCTION void operator()(const int thread_id, value_type& data) const
  {
    printf(" thread_id = %d\n", thread_id);
    CheckSPOSteps<Devices::KOKKOS>::thread_main(ncrews_,
                                                thread_id,
                                                team_size_,
                                                ions_,
                                                spo_main_,
                                                spo_ref_main_,
                                                nsteps_,
                                                Rmax_,
                                                data.ratio,
                                                data.nspheremoves,
                                                data.dNumVGHCalls,
                                                data.evalV_v_err,
                                                data.evalVGH_v_err,
                                                data.evalVGH_g_err,
                                                data.evalVGH_h_err);
  }

  KOKKOS_INLINE_FUNCTION void join(volatile value_type& dst, const volatile value_type& src) const { dst += src; }

  KOKKOS_INLINE_FUNCTION void init(value_type& dst) const { dst = {0, 0, 0, 0, 0, 0, 0}; }

private:
  int ncrews_;
  int crewsize_;
  int team_size_;
  ParticleSet ions_;
  SPODevImp spo_main_;
  SPORef spo_ref_main_;
  int nsteps_;
  QMCT::RealType Rmax_;
};
    	    
} // namespace qmcplusplus

#endif

