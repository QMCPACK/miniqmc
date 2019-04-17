#include "Drivers/CheckSPOData.hpp"
#include "Drivers/CheckSPOSteps.hpp"

namespace qmcplusplus
{

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
    
template<>
void CheckSPOSteps<Devices::KOKKOS>::finalize()
{
  Kokkos::finalize();
}

template<>
void CheckSPOSteps<Devices::KOKKOS>::initialize(int argc, char** argv)
{
  std::cout << "CheckSPOSteps<DDT::KOKKOS>::initialize" << '\n';
  Kokkos::initialize(argc, argv);
}

template<>
template<typename T>
CheckSPOData<T> CheckSPOSteps<Devices::KOKKOS>::runThreads(const int team_size,
							   ParticleSet& ions,
							   const SPODevImp& spo_main,
							   const SPORef& spo_ref_main,
							   const int nsteps,
							   const T Rmax)
{
  CheckSPOData<T> my_data{0, 0, 0, 0, 0, 0, 0};
  int num_threads = 1;
#ifdef KOKKOS_ENABLE_OPENMP
    num_threads = 	Kokkos::OpenMP::thread_pool_size();
#endif


    int ncrews      = num_threads;
  int crewsize    = 1;
  //Its my belieif this is what the CPU implementation does
  printf(" In partition master with %d threads, %d crews.  team_size = %d \n", num_threads, ncrews, team_size);
  

#if defined(KOKKOS_ENABLE_OPENMP) && !defined(KOKKOS_ENABLE_CUDA)
  // The kokkos check_determinant was never threaded
  // could be with
  //
  Kokkos::parallel_reduce(team_size,
			  SPOReduction<T>(team_size,
				       ions,
				       spo_main,
				       spo_ref_main,
				       nsteps,
				       Rmax),
			  my_data);
  //Kokkos::OpenMP::partition_master(main_function,nmovers,crewsize);
#else
      CheckSPOSteps<Devices::KOKKOS>::thread_main(1,
                                                0,
                                                1,
                                                ions,
                                                spo_main,
                                                spo_ref_main,
                                                nsteps,
                                                Rmax,
                                                my_data.ratio,
                                                my_data.nspheremoves,
                                                my_data.dNumVGHCalls,
						my_data.evalV_v_err,
                                                my_data.evalVGH_v_err,
                                                my_data.evalVGH_g_err,
						my_data.evalVGH_h_err);
#endif
  return my_data;
}

// extern template CheckSPOData<double> CheckSPOSteps<Devices::KOKKOS>::runThreads(const int team_size,
// 							   ParticleSet& ions,
// 							   const SPODevImp& spo_main,
// 							   const SPORef& spo_ref_main,
// 							   const int nsteps,
// 										const double Rmax);

// extern template CheckSPOData<float> CheckSPOSteps<Devices::KOKKOS>::runThreads(const int team_size,
// 							   ParticleSet& ions,
// 							   const SPODevImp& spo_main,
// 							   const SPORef& spo_ref_main,
// 							   const int nsteps,
// 										const float Rmax);

//extern template void CheckSPOSteps<Devices::KOKKOS>::finalize();
//extern template void CheckSPOSteps<Devices::KOKKOS>::initialize(int argc, char** argv);

} // namespace qmcplusplus

#include "Drivers/CheckSPOStepsKOKKOS.hpp"

  namespace qmcplusplus
  {
  template void CheckSPOSteps<Devices::KOKKOS>::test(int&, int, qmcplusplus::Tensor<int, 3u> const&, int, int, int, int, int, double);

  template class CheckSPOSteps<Devices::KOKKOS>;

  }
