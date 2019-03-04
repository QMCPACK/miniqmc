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

#ifndef QMCPLUSPLUS_CHECK_SPO_H
#define QMCPLUSPLUS_CHECK_SPO_H
#include <Utilities/Configuration.h>
#include <Utilities/Communicate.h>
#include <Particle/ParticleSet.h>
#include <Particle/ParticleSet_builder.hpp>
#include <Utilities/RandomGenerator.h>
#include <Input/Input.hpp>
#include <QMCWaveFunctions/EinsplineSPO.hpp>
#include "QMCWaveFunctions/EinsplineSPODevice.hpp"
#include "QMCWaveFunctions/EinsplineSPODeviceImp.hpp"
#include <QMCWaveFunctions/einspline_spo_ref.hpp>
#include <Utilities/qmcpack_version.h>
#include <getopt.h>


namespace qmcplusplus
{
class CheckSPOTest
{
public:
  using QMCT = QMCTraits;
  // clang-format off
  typedef QMCTraits::RealType           RealType;
  typedef ParticleSet::ParticlePos_t    ParticlePos_t;
  typedef ParticleSet::PosType          PosType;
  // clang-format on

  // use the global generator

  int na              = 1;
  int nb              = 1;
  int nc              = 1;
  int nsteps          = 5;
  int iseed           = 11;
  QMCT::RealType Rmax = 1.7;
  int nx = 37, ny = 37, nz = 37;
  // thread blocking
  // int team_size=1; //default is 1
  int tileSize  = -1;
  int team_size = 1;
  bool verbose  = false;

  Tensor<int, 3> tmat;


  ParticleSet ions;
  Tensor<OHMMS_PRECISION, 3> lattice_b;
  int error;

  void printHelp();

  void setup(int argc, char** argv);

  int runTests();
};

template<typename T>
struct CheckSPOData
{
  T ratio;
  T nspheremoves;
  T dNumVGHCalls;
  T evalV_v_err;
  T evalVGH_v_err;
  T evalVGH_g_err;
  T evalVGH_h_err;

  volatile CheckSPOData& operator+=(const volatile CheckSPOData& data) volatile
  {
    ratio += data.ratio;
    nspheremoves += data.nspheremoves;
    dNumVGHCalls += data.dNumVGHCalls;
    evalV_v_err += data.evalV_v_err;
    evalVGH_v_err += data.evalVGH_v_err;
    evalVGH_g_err += data.evalVGH_g_err;
    evalVGH_h_err += data.evalVGH_h_err;
    return *this;
  }
};


template<Devices DT>
class CheckSPOSteps
{
public:
  using SPODevImp = EinsplineSPO<DT, OHMMS_PRECISION>;
  using SPORef    = miniqmcreference::EinsplineSPO_ref<OHMMS_PRECISION>;

public:
  using QMCT = QMCTraits;
  static void initialize(int arc, char** argv);
  static void test(int& error,
                   int team_size,
                   const Tensor<int, 3>& tmat,
                   int tileSize,
                   const int nx,
                   const int ny,
                   const int nz,
                   const int nsteps,
                   const QMCT::RealType Rmax);
  static void finalize();

  template<typename T>
  static void thread_main(const int num_threads,
                          const int thread_id,
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
                          T& evalVGH_h_err);
private:
  static SPODevImp buildSPOMain(const int nx,
			 const int ny,
			 const int nz,
			 const int norb,
			 const int nTiles,
			 const Tensor<OHMMS_PRECISION, 3>& lattice_b);

  template<typename T>
  static CheckSPOData<T> runThreads(const int team_size,
                                    ParticleSet& ions,
                                    const SPODevImp& spo_main,
                                    const SPORef& spo_ref_main,
                                    const int nsteps,
                                    const T Rmax);


};
} // namespace qmcplusplus
#ifdef QMC_USE_KOKKOS
#include "Drivers/test/CheckSPOStepsKOKKOS.hpp"
#endif
// #ifdef QMC_USE_CUDA
// #include "Drivers/test/CheckSPOStepsCUDA.hpp"
// #endif

#endif
