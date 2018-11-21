#include "Devices.h"
#include "Drivers/MiniqmcDriverFunctions.hpp"
#include "Drivers/MiniqmcDriver.h"
#include <boost/hana/lazy.hpp>
#include <boost/hana/eval.hpp>
#include <boost/hana/length.hpp>
#include <boost/hana/equal.hpp>
#include <boost/hana/find.hpp>
#include <boost/hana/assert.hpp>
//#include <boost/hana/experimental/printable.hpp>
namespace qmcplusplus
{

void MiniqmcDriver::initialize(int argc, char** argv)
{
  // using init_func       = std::function<void(int argc, char** argv)>;
  // auto initializeCPU    = std::bind(&MiniqmcDriverFunctions<Devices::CPU>::initialize,
  // 				    std::placeholders::_1,
  // 				    std::placeholders::_2);
  // auto initializeKOKKOS = std::bind(&MiniqmcDriverFunctions<Devices::KOKKOS>::initialize,
  //                                   std::placeholders::_1,
  //                                   std::placeholders::_2);

  //using DEVICE_TYPE = decltype(mdf_map[device]);
  //auto& MDFI = mdf_map[device];
  MiniqmcDriverFunctions<Devices::KOKKOS>::initialize(argc, argv);
  comm = new Communicate(argc, argv);

  if (!comm->root())
  {
    outputManager.shutOff();
  }

  int number_of_electrons = 0;

  Tensor<int, 3> tmat(mq_opt_.na, 0, 0, 0, mq_opt_.nb, 0, 0, 0, mq_opt_.nc);

  timer_levels timer_level = timer_level_fine;
  if (mq_opt_.timer_level_name == "coarse")
  {
    timer_level = timer_level_coarse;
  }
  else if (mq_opt_.timer_level_name != "fine")
  {
    app_error() << "Timer level should be 'coarse' or 'fine', name given: "
                << mq_opt_.timer_level_name << '\n';
    return; //should throw
  }

  TimerManager.set_timer_threshold(timer_level);
  setup_timers(mq_opt_.Timers, mq_opt_.MiniQMCTimerNames, timer_level_coarse);

  if (comm->root())
  {
    if (mq_opt_.verbose)
      outputManager.setVerbosity(Verbosity::HIGH);
    else
      outputManager.setVerbosity(Verbosity::LOW);
  }

  print_version(mq_opt_.verbose);

  int nTiles = 1;

  // initialize ions and splines which are shared by all threads later
  {
    Tensor<OHMMS_PRECISION, 3> lattice_b;
    build_ions(ions, tmat, lattice_b);
    const int nels   = count_electrons(ions, 1);
    const int norb   = nels / 2;
    mq_opt_.tileSize = (mq_opt_.tileSize > 0) ? mq_opt_.tileSize : norb;
    nTiles           = norb / mq_opt_.tileSize;

    number_of_electrons = nels;

    const size_t SPO_coeff_size = static_cast<size_t>(norb) * (mq_opt_.nx + 3) * (mq_opt_.ny + 3) *
        (mq_opt_.nz + 3) * sizeof(QMCT::RealType);
    const double SPO_coeff_size_MB = SPO_coeff_size * 1.0 / 1024 / 1024;

    app_summary() << "Number of orbitals/splines = " << norb << '\n'
                  << "Tile size = " << mq_opt_.tileSize << '\n'
                  << "Number of tiles = " << nTiles << '\n'
                  << "Number of electrons = " << nels << '\n'
                  << "Rmax = " << mq_opt_.Rmax << '\n';
    app_summary() << "Iterations = " << mq_opt_.nsteps << '\n';
    app_summary() << "OpenMP threads = " << omp_get_max_threads() << '\n';
#ifdef HAVE_MPI
    app_summary() << "MPI processes = " << comm.size() << '\n';
#endif

    app_summary() << "\nSPO coefficients size = " << SPO_coeff_size << " bytes ("
                  << SPO_coeff_size_MB << " MB)" << '\n';

    spo_main =
        build_SPOSet(mq_opt_.useRef, mq_opt_.nx, mq_opt_.ny, mq_opt_.nz, norb, nTiles, lattice_b);
  }

  if (!mq_opt_.useRef)
    app_summary() << "Using SoA distance table, Jastrow + einspline, " << '\n'
                  << "and determinant update." << '\n'
		  << "with " << mq_opt_.device_number << " device implementation" << '\n';
  else
    app_summary() << "Using the reference implementation for Jastrow, " << '\n'
                  << "determinant update, and distance table + einspline of the " << '\n'
                  << "reference implementation " << '\n';

  mq_opt_.Timers[mq_opt_.Timer_Total]->start();


  //Now lets figure out what threading sizes are needed:
  //  For walker level parallelism:
}


void MiniqmcDriver::run()
{
    //    decltype(std::declval<device_map[mq_opt_.device_number]>)::runThreads(mq_opt_, myPrimes, ions, spo_main);

  MiniqmcDriverFunctions<Devices::CPU> myCPU;
  constexpr auto device_map =
  hana::make_map(
		hana::make_pair(hana::int_c<static_cast<int>(Devices::CPU)>,
				hana::type_c<MiniqmcDriverFunctions<Devices::CPU>>),
#ifdef QMC_USE_KOKKOS
		hana::make_pair(hana::int_c<static_cast<int>(Devices::KOKKOS)>,
				hana::type_c<MiniqmcDriverFunctions<Devices::KOKKOS>>),
#endif
#ifdef QMC_USE_OMPOL
		hana::make_pair(hana::int_c<static_cast<int>(Devices::OMPOL)>,
				hana::type_c<MiniqmcDriverFunctions<Devices::OMPOL>>),
#endif
		hana::make_pair(hana::int_c<static_cast<int>(Devices::LAST)>,
				hana::type_c<MiniqmcDriverFunctions<Devices::CPU>>)
				   );

  //hana::print(hana::size_c<static_cast<size_t>(Devices::LAST)>);
  //BOOST_HANA_CHECK(hana::length(device_tuple) == hana::size_c<hana::int_c<static_cast<size_t>(Devices::LAST)>>);
  BOOST_HANA_CHECK(hana::size_c<hana::int_c<3>> == hana::length(device_tuple));
  BOOST_HANA_CHECK(hana::size_c<hana::int_c<2>> == hana::size_c<static_cast<size_t>(Devices::LAST)>);
  //BOOST_HANA_CHECK((hana::size_c<static_cast<size_t>(Devices::LAST)> == hana::length(device_tuple));
  
  //static_assert(device_map[hana::int_c<0>] == hana::type_c<MiniqmcDriverFunctions<Devices::CPU>>,"");
  //auto type_wrapped_devices_range = hana::transform(devices_range, hana::make_type);
  using MyHandler = decltype(hana::unpack(devices_range, hana::template_<CaseHandler>))::type;
  
  MyHandler handler(*this);

  handler.handle(mq_opt_.device_number);
  
  // switch (mq_opt_.device_number)
  // {
  // case 0:
  //   using df = decltype(+device_tuple[hana::size_c<0>])::type;
  //   decltype(+device_tuple[hana::size_c<0>])::type::runThreads(mq_opt_, myPrimes, ions, spo_main);
  //   //MiniqmcDriverFunctions<Devices::CPU>::runThreads(mq_opt_, myPrimes, ions, spo_main);
  //   break;
  // case 1:
  //   MiniqmcDriverFunctions<Devices::KOKKOS>::runThreads(mq_opt_, myPrimes, ions, spo_main);
  //   break;
  // default:
  //   break;
  // }
    results();
}

} // namespace qmcplusplus
