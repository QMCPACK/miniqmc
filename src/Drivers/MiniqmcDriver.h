#ifndef QMCPLUSPLUS_MINIQMC_DRIVER_H
#define QMCPLUSPLUS_MINIQMC_DRIVER_H

#include <boost/hana/for_each.hpp>
#include <boost/hana/map.hpp>
#include <boost/hana/type.hpp>
#include <boost/hana/adapt_struct.hpp>
#include <Utilities/Configuration.h>
#include <Utilities/Communicate.h>
#include <Particle/ParticleSet.h>
#include <Particle/DistanceTable.h>
#include <Utilities/PrimeNumberSet.h>
#include <Utilities/XMLWriter.h>
#include <Utilities/RandomGenerator.h>
#include <Utilities/qmcpack_version.h>
#include <Input/Input.hpp>
#include "Devices.h"
#include <QMCWaveFunctions/SPOSet.h>
#include <QMCWaveFunctions/SPOSet_builder.h>
#include <QMCWaveFunctions/WaveFunction.h>
#include <Drivers/Mover.hpp>
#include "Drivers/MiniqmcOptions.hpp"
#include "Drivers/MiniqmcDriverFunctions.hpp"

namespace qmcplusplus
{
namespace hana = boost::hana;

  constexpr auto device_tuple =
  hana::make_tuple(hana::type_c<MiniqmcDriverFunctions<Devices::CPU>>,
#ifdef QMC_USE_KOKKOS
		   hana::type_c<MiniqmcDriverFunctions<Devices::KOKKOS>>,
#endif
#ifdef QMC_USE_OMPOL
		   hana::type_c<MiniqmcDriverFunctions<Devices::OMPOL>>,
#endif
		   hana::type_c<MiniqmcDriverFunctions<Devices::CPU>>);

class MiniqmcDriver
{
public:
  template<typename ...DN>
  struct CaseHandler
  {
    template <typename ...> struct IntList {};

  void handle_cases(MiniqmcDriver& my_, int, IntList<>) { /* "default case" */ }

  template<typename ...N> void handle_cases(MiniqmcDriver& my_, int i)
  {
    handle_cases(my_, i, IntList<N...>());
  }

    template<typename I, typename ...N>
  void handle_cases(MiniqmcDriver& my_, int i, IntList<I, N...>)
  {
    if (I::value != i) { return handle_cases(my_, i, IntList<N...>()); }
    decltype(+device_tuple[hana::size_c<I::value>])::type::runThreads(my_.mq_opt_,
							       my_.myPrimes,
							       my_.ions,
							       my_.spo_main);
  }

    
    MiniqmcDriver& my_;
    CaseHandler(MiniqmcDriver& my): my_(my) {}
    void handle(int i)
    {
      handle_cases<DN...>(my_, i);
    }
  };





   //static constexpr auto mdfs = hana::make_map(hana::for_each(Devices, [&](auto x){
		    //								 return hana::pair(hana::type_c(x), hana::type_c(MiniqmcDriverFunctions<x>))});

  MiniqmcDriver(MiniqmcOptions mq_opt) : mq_opt_(mq_opt) {}

  void initialize(int argc, char** argv);
  
  void run();
  
  ~MiniqmcDriver()
  {
    if (spo_main != nullptr)
      delete spo_main;
    if (comm != nullptr)
      delete comm;
  }

private:
  void results()
  {
    if (comm->root())
    {
      std::cout << "================================== " << '\n';

      mq_opt_.Timers[mq_opt_.Timer_Total]->stop();
      TimerManager.print();

      XMLDocument doc;
      XMLNode* resources = doc.NewElement("resources");
      XMLNode* hardware  = doc.NewElement("hardware");
      resources->InsertEndChild(hardware);
      doc.InsertEndChild(resources);
      XMLNode* timing = TimerManager.output_timing(doc);
      resources->InsertEndChild(timing);

      XMLNode* particle_info = doc.NewElement("particles");
      resources->InsertEndChild(particle_info);
      XMLNode* electron_info = doc.NewElement("particle");
      electron_info->InsertEndChild(MakeTextElement(doc, "name", "e"));
      electron_info->InsertEndChild(
          MakeTextElement(doc, "size", std::to_string(count_electrons(ions, 1))));
      particle_info->InsertEndChild(electron_info);


      XMLNode* run_info    = doc.NewElement("run");
      XMLNode* driver_info = doc.NewElement("driver");
      driver_info->InsertEndChild(MakeTextElement(doc, "name", "miniqmc"));
      driver_info->InsertEndChild(MakeTextElement(doc, "steps", std::to_string(mq_opt_.nsteps)));
      driver_info->InsertEndChild(MakeTextElement(doc, "substeps", std::to_string(mq_opt_.nsubsteps)));
      run_info->InsertEndChild(driver_info);
      resources->InsertEndChild(run_info);

      std::string info_name = "info_" + std::to_string(mq_opt_.na) + "_" +
          std::to_string(mq_opt_.nb) + "_" + std::to_string(mq_opt_.nc) + ".xml";
      doc.SaveFile(info_name.c_str());
    }
  }

  
  void main_function()
  {
    

  }

  using QMCT = QMCTraits;
  // clang-format off
  typedef ParticleSet::ParticlePos_t    ParticlePos_t;
  typedef ParticleSet::PosType          PosType;
  // clang-format on
  // use the global generator

  MiniqmcOptions mq_opt_;
  // int device_number = 0;
  // int na     = 1;
  // int nb     = 1;
  // int nc     = 1;
  // int nsteps = 5;
  // int iseed  = 11;
  // int nx = 37, ny = 37, nz = 37;
  // int nmovers = 1;
  // // thread blocking
  // int tileSize  = -1;
  // int team_size = 1;
  // int nsubsteps = 1;
  // int ncrews = 1;
  // // Set cutoff for NLPP use.
  // QMCT::RealType Rmax = 1.7;
  // bool useRef   = false;
  // bool enableJ3 = false;
  // bool verbose                 = false;
  // std::string timer_level_name = "fine";

  PrimeNumberSet<uint32_t> myPrimes;
  ParticleSet ions;
  SPOSet* spo_main = nullptr;
  Communicate* comm = nullptr;
};

} // namespace qmcplusplus

#endif
