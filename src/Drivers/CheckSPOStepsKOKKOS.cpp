#include "Drivers/CheckSPOData.hpp"
#include "Drivers/CheckSPOStepsKOKKOS.hpp"

namespace qmcplusplus
{
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
  num_threads = Kokkos::OpenMP::thread_pool_size();
#endif


  int ncrews   = num_threads;
  int crewsize = 1;
  //Its my belieif this is what the CPU implementation does
  printf(" In partition master with %d threads, %d crews.  team_size = %d \n", num_threads, ncrews, team_size);


#if defined(KOKKOS_ENABLE_OPENMP) && !defined(KOKKOS_ENABLE_CUDA)
  // The kokkos check_determinant was never threaded
  // could be with
  //
  Kokkos::parallel_reduce(team_size, SPOReduction<T>(team_size, ions, spo_main, spo_ref_main, nsteps, Rmax), my_data);
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


namespace qmcplusplus
{
template class CheckSPOSteps<Devices::KOKKOS>;
} // namespace qmcplusplus
