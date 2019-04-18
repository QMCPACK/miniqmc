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
#include "Devices_HANA.hpp"
#include <QMCWaveFunctions/SPOSet.h>
#include <QMCWaveFunctions/SPOSet_builder.h>
#include <QMCWaveFunctions/WaveFunction.h>
#include <Drivers/Mover.hpp>
#include "Drivers/MiniqmcOptions.hpp"
#include "Drivers/MiniqmcDriverFunctions.hpp"
//#include "Drivers/getMiniQMCDriverFunctions.h"

/** @file
 *  This contains the machinery for the static dispatch of DriverFunctions for 
 *  different devices. The metaprogramming here could probably be inproved significantly,
 *  but should seldom need to be updated.  
 */

namespace qmcplusplus
{
namespace hana = boost::hana;

/** This creates a compile time tuple of the different devices types, indexed by the device range
 */
constexpr auto device_tuple = hana::make_tuple(hana::type_c<MiniqmcDriverFunctions<Devices::CPU>>,
#ifdef QMC_USE_KOKKOS
                                               hana::type_c<MiniqmcDriverFunctions<Devices::KOKKOS>>,
#endif
#ifdef QMC_USE_OMPOL
                                               hana::type_c<MiniqmcDriverFunctions<Devices::OMPOL>>,
#endif
#ifdef QMC_USE_CUDA
                                               hana::type_c<MiniqmcDriverFunctions<Devices::CUDA>>,
#endif

                                               hana::type_c<MiniqmcDriverFunctions<Devices::CPU>>);
/// The final type is so the tuple and device enum have the same length, this is important for the for each device test code.


/** Owns the comm and Main SPOSet. Handles dispatch to correct device driver functions.
 *  
 *  contains the boiler plate to handle the dispatching to 
 *  appropriate functions from MiniqmcDriverFunctions class templates.
 */
class MiniqmcDriver
{
public:
  /** template functions create the equivalent to a select containing each of the devices.
   *  Right now one set of cases is required for each function. Perhaps this can be generalized
   *  using std::forward and the like.
   */
  template<typename... DN>
  struct CaseHandler
  {
    template<typename...>
    struct IntList
    {};

    /** "default case" */
    void build_cases(SPOSet*& spo_set,
                     MiniqmcOptions& mq_opt,
                     const int norb,
                     const int nTiles,
                     const int splines_per_block,
                     const Tensor<OHMMS_PRECISION, 3>& lattice_b,
                     int i,
                     IntList<>)
    {}

    template<typename... N>
    void build_cases(SPOSet*& spo_set,
                     MiniqmcOptions& mq_opt,
                     const int norb,
                     const int nTiles,
                     const int splines_per_block,
                     const Tensor<OHMMS_PRECISION, 3>& lattice_b,
                     int i)
    {
      build_cases(spo_set, mq_opt, norb, nTiles, splines_per_block, lattice_b, i, IntList<N...>());
    }

    template<typename I, typename... N>
    void build_cases(SPOSet*& spo_set,
                     MiniqmcOptions& mq_opt,
                     const int norb,
                     const int nTiles,
                     const int splines_per_block,
                     const Tensor<OHMMS_PRECISION, 3>& lattice_b,
                     int i,
                     IntList<I, N...>)
    {
      if (I::value != i)
      {
        return build_cases(spo_set, mq_opt, norb, nTiles, splines_per_block, lattice_b, i, IntList<N...>());
      }
      decltype(+device_tuple[hana::size_c<I::value>])::type::buildSPOSet(spo_set,
                                                                         mq_opt,
                                                                         norb,
                                                                         nTiles,
                                                                         splines_per_block,
                                                                         lattice_b);
    }

    void run_cases(MiniqmcDriver& my_, int, IntList<>) {}

    template<typename... N>
    void run_cases(MiniqmcDriver& my_, int i)
    {
      run_cases(my_, i, IntList<N...>());
    }

    template<typename I, typename... N>
    void run_cases(MiniqmcDriver& my_, int i, IntList<I, N...>)
    {
      if (I::value != i)
      {
        return run_cases(my_, i, IntList<N...>());
      }
      if (my_.mq_opt_.enableCrowd)
      {
        decltype(+device_tuple[hana::size_c<I::value>])::type::movers_runThreads(my_.mq_opt_,
                                                                                 my_.myPrimes,
                                                                                 my_.ions,
                                                                                 my_.spo_main);
      }
      else
      {
        decltype(
            +device_tuple[hana::size_c<I::value>])::type::runThreads(my_.mq_opt_, my_.myPrimes, my_.ions, my_.spo_main);
      }
    }

    void initialize_cases(int argc, char** argv, int, IntList<>)
    { /* "default case" */
    }

    template<typename... N>
    void initialize_cases(int argc, char** argv, int i)
    {
      initialize_cases(argc, argv, i, IntList<N...>());
    }

    template<typename I, typename... N>
    void initialize_cases(int argc, char** argv, int i, IntList<I, N...>)
    {
      if (I::value != i)
      {
        return initialize_cases(argc, argv, i, IntList<N...>());
      }
      decltype(+device_tuple[hana::size_c<I::value>])::type::initialize(argc, argv);
    }


    /** build the main sposet for the correct device. 
     *  build_cases will expand to handle all the built devices.
     */
    MiniqmcDriver& my_;
    CaseHandler(MiniqmcDriver& my) : my_(my) {}
    void build(SPOSet*& spo_set,
               MiniqmcOptions& mq_opt,
               const int norb,
               const int nTiles,
               const int splines_per_block,
               const Tensor<OHMMS_PRECISION, 3>& lattice_b,
               int i)
    {
      build_cases<DN...>(spo_set, mq_opt, norb, nTiles, splines_per_block, lattice_b, i);
    }

    /** run_cases will expand to handle all the built devices. */
    void run(int i) { run_cases<DN...>(my_, i); }

    /** In case the device or framework needs its runtime initialized once before threading.
     *  initialize_cases will expand to handle all build devices
     */
    void initialize(int argc, char** argv, int i) { initialize_cases<DN...>(argc, argv, i); }
  };

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

      mq_opt_.Timers[Timer_Total]->stop();

      std::cout << '\n' << "========== Throughput ============ " << '\n' << '\n';
      std::cout << "Total throughput ( N_walkers * N_elec^3 / Total time ) = "
		<< (mq_opt_.nmovers * mq_opt_.pack_size * comm->size() * std::pow(double(mq_opt_.nels),3) / mq_opt_.Timers[Timer_Total]->get_total()) << '\n';
      std::cout << "Diffusion throughput ( N_walkers * N_elec^3 / Diffusion time ) = "
         << (mq_opt_.nmovers * mq_opt_.pack_size * comm->size() * std::pow(double(mq_opt_.nels),3) / mq_opt_.Timers[Timer_Diffusion]->get_total()) << '\n';
      std::cout << "Pseudopotential throughput ( N_walkers * N_elec^2 / Pseudopotential time ) = "
         << (mq_opt_.nmovers * mq_opt_.pack_size * comm->size() * std::pow(double(mq_opt_.nels),2) / mq_opt_.Timers[Timer_ECP]->get_total()) << '\n';
      std::cout << std::endl;
    
      TimerManagerClass::get().print();

      XMLDocument doc;
      XMLNode* resources = doc.NewElement("resources");
      XMLNode* hardware  = doc.NewElement("hardware");
      resources->InsertEndChild(hardware);
      doc.InsertEndChild(resources);
      XMLNode* timing = TimerManagerClass::get().output_timing(doc);
      resources->InsertEndChild(timing);

      XMLNode* particle_info = doc.NewElement("particles");
      resources->InsertEndChild(particle_info);
      XMLNode* electron_info = doc.NewElement("particle");
      electron_info->InsertEndChild(MakeTextElement(doc, "name", "e"));
      electron_info->InsertEndChild(MakeTextElement(doc, "size", std::to_string(count_electrons(ions, 1))));
      particle_info->InsertEndChild(electron_info);

      XMLNode* run_info    = doc.NewElement("run");
      XMLNode* driver_info = doc.NewElement("driver");
      driver_info->InsertEndChild(MakeTextElement(doc, "name", "miniqmc"));
      driver_info->InsertEndChild(MakeTextElement(doc, "steps", std::to_string(mq_opt_.nsteps)));
      driver_info->InsertEndChild(MakeTextElement(doc, "substeps", std::to_string(mq_opt_.nsubsteps)));
      run_info->InsertEndChild(driver_info);
      resources->InsertEndChild(run_info);

      std::string info_name = "info_" + std::to_string(mq_opt_.na) + "_" + std::to_string(mq_opt_.nb) + "_" +
          std::to_string(mq_opt_.nc) + ".xml";
      doc.SaveFile(info_name.c_str());
    }
  }

  /** Why? */
  void main_function() {}

  using QMCT = QMCTraits;
  typedef ParticleSet::ParticlePos_t ParticlePos_t;
  typedef ParticleSet::PosType PosType;

  MiniqmcOptions mq_opt_;
  PrimeNumberSet<uint32_t> myPrimes;
  ParticleSet ions;
  SPOSet* spo_main  = nullptr;
  Communicate* comm = nullptr;
};

} // namespace qmcplusplus

#endif
