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
#include "Drivers/CheckSPOSteps.hpp"
#include "Drivers/CheckSPOData.hpp"
#include <Utilities/qmcpack_version.h>
#include <getopt.h>
#include "Devices.h"
#include "Devices_HANA.hpp"


namespace qmcplusplus
{
namespace hana = boost::hana;

// forward declaration of CheckSPOSteps
template<Devices DT>
class CheckSPOSteps;
// This would be nice to generate with metaprogramming when there is time
constexpr auto device_spo_tuple = hana::make_tuple(hana::type_c<CheckSPOSteps<Devices::CPU>>,
#ifdef QMC_USE_KOKKOS
                                                   hana::type_c<CheckSPOSteps<Devices::KOKKOS>>,
#endif
#ifdef QMC_USE_OMPOL
                                                   hana::type_c<CheckSPOSteps<Devices::OMPOL>>,
#endif
#ifdef QMC_USE_CUDA
                                                   hana::type_c<CheckSPOSteps<Devices::CUDA>>,
#endif

                                                   hana::type_c<CheckSPOSteps<Devices::CPU>>);
// The final type is so the tuple and device enum have the same length, forget why this matters.

class CheckSPOTest
{
public:
  template<typename... DN>
  struct CaseHandler
  {
    template<typename...>
    struct IntList
    {};

    void test_cases(int& error,
                    const int team_size,
                    const Tensor<int, 3>& tmat,
                    int tileSize,
                    const int nx,
                    const int ny,
                    const int nz,
                    const int nsteps,
                    const double Rmax,
                    int i,
                    IntList<>)
    {}

    template<typename... N>
    void test_cases(int& error,
                    const int team_size,
                    const Tensor<int, 3>& tmat,
                    int tileSize,
                    const int nx,
                    const int ny,
                    const int nz,
                    const int nsteps,
                    const double Rmax,
                    int i)
    {
      test_cases(error, team_size, tmat, tileSize, nx, ny, nz, nsteps, Rmax, i, IntList<N...>());
    }

    template<typename I, typename... N>
    void test_cases(int& error,
                    const int team_size,
                    const Tensor<int, 3>& tmat,
                    int tileSize,
                    const int nx,
                    const int ny,
                    const int nz,
                    const int nsteps,
                    const double Rmax,
                    int i,
                    IntList<I, N...>)
    {
      if (I::value != i)
      {
        return test_cases(error, team_size, tmat, tileSize, nx, ny, nz, nsteps, Rmax, i, IntList<N...>());
      }
      decltype(+device_spo_tuple[hana::size_c<I::value>])::type::test(error,
                                                                      team_size,
                                                                      tmat,
                                                                      tileSize,
                                                                      nx,
                                                                      ny,
                                                                      nz,
                                                                      nsteps,
                                                                      Rmax);
    }
    CheckSPOTest& my_;
    CaseHandler(CheckSPOTest& my) : my_(my) {}

    void test(int& error,
              const int team_size,
              const Tensor<int, 3>& tmat,
              int tileSize,
              const int nx,
              const int ny,
              const int nz,
              const int nsteps,
              const double Rmax,
              int i)
    {
      test_cases<DN...>(error, team_size, tmat, tileSize, nx, ny, nz, nsteps, Rmax, i);
    }
  };

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
  int device          = 0;
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


} // namespace qmcplusplus
#endif
