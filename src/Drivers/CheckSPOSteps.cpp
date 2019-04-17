////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2019 QMCPACK developers.
//
// File developed by:
// Peter Doak, doakpw@ornl.gov, Oak Ridge National Lab
//
// File created by:
// Peter Doak, doakpw@ornl.gov, Oak Ridge National Lab
////////////////////////////////////////////////////////////////////////////////
#include <vector>
#include "Numerics/OhmmsPETE/Tensor.h"
#include "Drivers/CheckSPOSteps.hpp"

namespace qmcplusplus
{
template<Devices DT>
typename CheckSPOSteps<DT>::SPODevImp CheckSPOSteps<DT>::buildSPOMain(const int nx,
                                                                      const int ny,
                                                                      const int nz,
                                                                      const int norb,
                                                                      const int nTiles,
                                                                      const int splines_per_block,
                                                                      const Tensor<OHMMS_PRECISION, 3>& lattice_b)
{
  SPODevImp spo_main;
  spo_main.set(nx, ny, nz, norb, nTiles, splines_per_block);
  spo_main.setLattice(lattice_b);
  return spo_main;
}

template<Devices DT>
void CheckSPOSteps<DT>::test(int& error,
                             const int team_size,
                             const Tensor<int, 3>& tmat,
                             int tileSize,
                             const int nx,
                             const int ny,
                             const int nz,
                             const int nsteps,
                             const double Rmax)
{
  std::string enum_name = device_names[hana::int_c<static_cast<int>(DT)>];
  std::cout << "Testing Determinant Device Implementation: " << enum_name << '\n';

  ParticleSet ions;
  Tensor<OHMMS_PRECISION, 3> lattice_b;
  build_ions(ions, tmat, lattice_b);
  const int norb = count_electrons(ions, 1) / 2;
  tileSize       = (tileSize > 0) ? tileSize : norb;
  int nTiles     = norb / tileSize;
  if (norb > tileSize && norb % tileSize)
    ++nTiles;

  const size_t SPO_coeff_size    = static_cast<size_t>(norb) * (nx + 3) * (ny + 3) * (nz + 3) * sizeof(QMCT::RealType);
  const double SPO_coeff_size_MB = SPO_coeff_size * 1.0 / 1024 / 1024;

  app_summary() << "Number of orbitals/splines = " << norb << '\n'
                << "Tile size = " << tileSize << '\n'
                << "Number of tiles = " << nTiles << '\n'
                << "Rmax = " << Rmax << '\n';
  app_summary() << "Iterations = " << nsteps << '\n';
  app_summary() << "OpenMP threads = " << omp_get_max_threads() << '\n';

  app_summary() << "\nSPO coefficients size = " << SPO_coeff_size << " bytes (" << SPO_coeff_size_MB << " MB)" << '\n';

  SPODevImp spo_main = buildSPOMain(nx, ny, nz, norb, nTiles, tileSize, lattice_b);
  SPORef spo_ref_main;
  spo_ref_main.set(nx, ny, nz, norb, nTiles);
  spo_ref_main.Lattice.set(lattice_b);

  CheckSPOData<OHMMS_PRECISION> check_data =
      CheckSPOSteps::runThreads(team_size, ions, spo_main, spo_ref_main, nsteps, Rmax);

  OutputManagerClass::get().resume();

  check_data.evalV_v_err /= check_data.nspheremoves;
  check_data.evalVGH_v_err /= check_data.dNumVGHCalls;
  check_data.evalVGH_g_err /= check_data.dNumVGHCalls;
  check_data.evalVGH_h_err /= check_data.dNumVGHCalls;

  int np                           = omp_get_max_threads();
  constexpr QMCT::RealType small_v = std::numeric_limits<QMCT::RealType>::epsilon() * 1e4;
  constexpr QMCT::RealType small_g = std::numeric_limits<QMCT::RealType>::epsilon() * 3e6;
  constexpr QMCT::RealType small_h = std::numeric_limits<QMCT::RealType>::epsilon() * 6e8;
  int nfail                        = 0;
  app_log() << std::endl;
  if (check_data.evalV_v_err / np > small_v)
  {
    app_log() << "Fail in evaluate_v, V error =" << check_data.evalV_v_err / np << std::endl;
    nfail = 1;
  }
  if (check_data.evalVGH_v_err / np > small_v)
  {
    app_log() << "Fail in evaluate_vgh, V error =" << check_data.evalVGH_v_err / np << std::endl;
    nfail += 1;
  }
  if (check_data.evalVGH_g_err / np > small_g)
  {
    app_log() << "Fail in evaluate_vgh, G error =" << check_data.evalVGH_g_err / np << std::endl;
    nfail += 1;
  }
  if (check_data.evalVGH_h_err / np > small_h)
  {
    app_log() << "Fail in evaluate_vgh, H error =" << check_data.evalVGH_h_err / np << std::endl;
    nfail += 1;
  }

  if (nfail == 0)
    app_log() << "All checks passed for spo" << std::endl;
}


template<Devices DT>
template<typename T>
CheckSPOData<T> CheckSPOSteps<DT>::runThreads(int team_size,
                                              ParticleSet& ions,
                                              const SPODevImp& spo_main,
                                              const SPORef& spo_ref_main,
                                              int nsteps,
                                              T Rmax)
{
  T ratio         = 0.0;
  T nspheremoves  = 0;
  T dNumVGHCalls  = 0;
  T evalV_v_err   = 0.0;
  T evalVGH_v_err = 0.0;
  T evalVGH_g_err = 0.0;
  T evalVGH_h_err = 0.0;

#pragma omp parallel reduction(+:ratio,nspheremoves,dNumVGHCalls) \
   reduction(+:evalV_v_err,evalVGH_v_err,evalVGH_g_err,evalVGH_h_err)
  {
    const int np = omp_get_num_threads();
    const int ip = omp_get_thread_num();
    CheckSPOSteps<DT>::thread_main(np,
                                   ip,
                                   team_size,
                                   ions,
                                   spo_main,
                                   spo_ref_main,
                                   nsteps,
                                   Rmax,
                                   ratio,
                                   nspheremoves,
                                   dNumVGHCalls,
                                   evalV_v_err,
                                   evalVGH_v_err,
                                   evalVGH_g_err,
                                   evalVGH_h_err);
  }
  return CheckSPOData<T>{ratio, nspheremoves, dNumVGHCalls, evalV_v_err, evalVGH_v_err, evalVGH_g_err, evalVGH_h_err};
}


/** The inside of the parallel reduction for test.
 *  Since kokkos could magically marshall over MPI
 *  we have to pass by copy even if that is inefficient
 *  for on node parallelism like here.
 */
template<Devices DT>
template<typename T>
void CheckSPOSteps<DT>::thread_main(const int np,
                                    const int ip,
                                    const int team_size,
                                    const ParticleSet ions,
                                    const SPODevImp spo_main,
                                    const SPORef spo_ref_main,
                                    const int nsteps,
                                    const QMCT::RealType Rmax,
                                    T& ratio,
                                    T& nspheremoves,
                                    T& dNumVGHCalls,
                                    T& evalV_v_err,
                                    T& evalVGH_v_err,
                                    T& evalVGH_g_err,
                                    T& evalVGH_h_err)
{
  const int team_id   = ip / team_size;
  const int member_id = ip % team_size;

  // create generator within the thread
  std::cout << "Calling RandomGenerator constructor with MakeSeed(team_id, np) " << team_id << ", " << np << '\n';
  RandomGenerator<QMCT::RealType> random_th(MakeSeed(team_id, np));

  ParticleSet els;
  build_els(els, ions, random_th);
  els.update();

  const int nions = ions.getTotalNum();
  const int nels  = els.getTotalNum();
  const int nels3 = 3 * nels;

  // create pseudopp
  NonLocalPP<OHMMS_PRECISION> ecp(random_th);
  // create spo per thread
  SPODevImp spo(spo_main, team_size, member_id);
  //SPODevImp& spo = *dynamic_cast<SPODevImp*>(SPOSetBuilder<DT>::buildView(false, spo_main, team_size, member_id));
  SPORef spo_ref(spo_ref_main, team_size, member_id);

  // use teams
  // if(team_size>1 && team_size>=nTiles ) spo.set_range(team_size,ip%team_size);

  // this is the cutoff from the non-local PP
  const int nknots(ecp.size());

  ParticleSet::ParticlePos_t delta(nels);
  ParticleSet::ParticlePos_t rOnSphere(nknots);

  QMCT::RealType sqrttau = 2.0;
  QMCT::RealType accept  = 0.5;

  std::vector<QMCT::RealType> ur(nels);
  random_th.generate_uniform(ur.data(), nels);
  const double zval = 1.0 * static_cast<double>(nels) / static_cast<double>(nions);

  int my_accepted = 0, my_vals = 0;

  EinsplineSPOParams<T> esp = spo.getParams();
  for (int mc = 0; mc < nsteps; ++mc)
  {
    random_th.generate_normal(&delta[0][0], nels3);
    random_th.generate_uniform(ur.data(), nels);
    for (int ib = 0; ib < esp.nBlocks; ib++)

      // VMC
      for (int iel = 0; iel < nels; ++iel)
      {
        QMCT::PosType pos = els.R[iel] + sqrttau * delta[iel];

        spo.evaluate_vgh(pos);
        spo_ref.evaluate_vgh(pos);
        // accumulate error
        for (int ib = 0; ib < esp.nBlocks; ib++)
          for (int n = 0; n < esp.nSplinesPerBlock; n++)
          {
            // value
            evalVGH_v_err += std::fabs(spo.getPsi(ib, n) - spo_ref.psi[ib][n]);
            // grad
            evalVGH_g_err += std::fabs(spo.getGrad(ib, n, 0) - spo_ref.grad[ib].data(0)[n]);
            evalVGH_g_err += std::fabs(spo.getGrad(ib, n, 1) - spo_ref.grad[ib].data(1)[n]);
            evalVGH_g_err += std::fabs(spo.getGrad(ib, n, 2) - spo_ref.grad[ib].data(2)[n]);
            // hess
            evalVGH_h_err += std::fabs(spo.getHess(ib, n, 0) - spo_ref.hess[ib].data(0)[n]);
            evalVGH_h_err += std::fabs(spo.getHess(ib, n, 1) - spo_ref.hess[ib].data(1)[n]);
            evalVGH_h_err += std::fabs(spo.getHess(ib, n, 2) - spo_ref.hess[ib].data(2)[n]);
            evalVGH_h_err += std::fabs(spo.getHess(ib, n, 3) - spo_ref.hess[ib].data(3)[n]);
            evalVGH_h_err += std::fabs(spo.getHess(ib, n, 4) - spo_ref.hess[ib].data(4)[n]);
            evalVGH_h_err += std::fabs(spo.getHess(ib, n, 5) - spo_ref.hess[ib].data(5)[n]);
          }
        if (ur[iel] < accept)
        {
          els.R[iel] = pos;
          my_accepted++;
        }
      }

    random_th.generate_uniform(ur.data(), nels);
    ecp.randomize(rOnSphere); // pick random sphere
    for (int iat = 0, kat = 0; iat < nions; ++iat)
    {
      const int nnF    = static_cast<int>(ur[kat++] * zval);
      QMCT::RealType r = Rmax * ur[kat++];
      auto centerP     = ions.R[iat];
      my_vals += (nnF * nknots);

      for (int nn = 0; nn < nnF; ++nn)
      {
        for (int k = 0; k < nknots; k++)
        {
          QMCT::PosType pos = centerP + r * rOnSphere[k];
          spo.evaluate_v(pos);
          spo_ref.evaluate_v(pos);
          // accumulate error
          for (int ib = 0; ib < esp.nBlocks; ib++)
            for (int n = 0; n < esp.nSplinesPerBlock; n++)
              evalV_v_err += std::fabs(spo.getPsi(ib, n) - spo_ref.psi[ib][n]);
        }
      } // els
    }   // ions

  } // steps.

  ratio += QMCT::RealType(my_accepted) / QMCT::RealType(nels * nsteps);
  nspheremoves += QMCT::RealType(my_vals) / QMCT::RealType(nsteps);
  dNumVGHCalls += nels;
}

template class CheckSPOSteps<Devices::CPU>;
//template void CheckSPOSteps<Devices::CPU>::initialize(int, char**);
//This is one way to deal with generic functions used by specialized class templates
//Anoother is to put them in the header.
#ifdef QMC_USE_CUDA
template void CheckSPOSteps<Devices::CUDA>::test(int&, int, Tensor<int, 3u> const&, int, int, int, int, int, double);
template typename CheckSPOSteps<Devices::CUDA>::SPODevImp CheckSPOSteps<Devices::CUDA>::buildSPOMain(
    const int nx,
    const int ny,
    const int nz,
    const int norb,
    const int nTiles,
    const int tile_size,
    const Tensor<OHMMS_PRECISION, 3>& lattice_b);
#endif

#ifdef QMC_USE_KOKKOS
template void CheckSPOSteps<Devices::KOKKOS>::test(int&, int, Tensor<int, 3u> const&, int, int, int, int, int, double);
template typename CheckSPOSteps<Devices::KOKKOS>::SPODevImp CheckSPOSteps<Devices::KOKKOS>::buildSPOMain(
    const int nx,
    const int ny,
    const int nz,
    const int norb,
    const int nTiles,
    const int tile_size,
    const Tensor<OHMMS_PRECISION, 3>& lattice_b);

#endif

} // namespace qmcplusplus
