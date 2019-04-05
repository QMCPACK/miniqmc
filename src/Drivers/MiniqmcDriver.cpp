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
  using MyHandler = decltype(hana::unpack(devices_range, hana::template_<CaseHandler>))::type;
  
  MyHandler handler(*this);

  handler.initialize(argc, argv, mq_opt_.device_number);
  
  comm = new Communicate(argc, argv);

  if (!comm->root())
  {
    OutputManagerClass::get().shutOff();
  }

  int number_of_electrons = 0;

  Tensor<int, 3> tmat(mq_opt_.na, 0, 0, 0, mq_opt_.nb, 0, 0, 0, mq_opt_.nc);

  timer_levels timer_level = timer_level_fine;
  // if (mq_opt_.timer_level_name == "coarse")
  // {
  //   timer_level = timer_level_coarse;
  // }
  // else if (mq_opt_.timer_level_name != "fine")
  // {
  //   app_error() << "Timer level should be 'coarse' or 'fine', name given: "
  //               << mq_opt_.timer_level_name << '\n';
  //   return; //should throw
  // }
  
  TimerManagerClass::get().set_timer_threshold(timer_level);
  setup_timers(mq_opt_.Timers, mq_opt_.MiniQMCTimerNames, timer_level_coarse);

  if (comm->root())
  {
    if (mq_opt_.verbose)
      OutputManagerClass::get().setVerbosity(Verbosity::HIGH);
    else
      OutputManagerClass::get().setVerbosity(Verbosity::LOW);
  }

  print_version(mq_opt_.verbose);

  int nTiles = 1;

  mq_opt_.Timers[Timer_Total]->start();
  mq_opt_.Timers[Timer_Init]->start();

  // initialize ions and splines which are shared by all threads later
  
    Tensor<OHMMS_PRECISION, 3> lattice_b;
    build_ions(ions, tmat, lattice_b);
    const int nels   = count_electrons(ions, 1);
    const int norb   = nels / 2;
    mq_opt_.splines_per_block = (mq_opt_.splines_per_block > 0) ? mq_opt_.splines_per_block : norb;
    nTiles           = norb / mq_opt_.splines_per_block;

    if ( norb > mq_opt_.splines_per_block || norb % mq_opt_.splines_per_block )
      ++nTiles;

    number_of_electrons = nels;

    const size_t SPO_coeff_size = static_cast<size_t>(norb) * (mq_opt_.nx + 3) * (mq_opt_.ny + 3) *
        (mq_opt_.nz + 3) * sizeof(QMCT::RealType);
    const double SPO_coeff_size_MB = SPO_coeff_size * 1.0 / 1024 / 1024;
  const size_t determinant_buffer_size = (static_cast<size_t>(norb) * norb + norb * 3) * sizeof(QMCT::RealType) * 2;
  const double determinant_buffer_size_MB =  determinant_buffer_size * 1.0 / 1024 / 1024;

    app_summary() << "Number of orbitals/splines = " << norb << '\n'
                  << "splines per block = " << mq_opt_.splines_per_block << '\n'
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
    app_summary() << "\nDeterminant Buffer size (per walker) = " << determinant_buffer_size << " bytes (" << determinant_buffer_size_MB << " MB)" << '\n';

    handler.build(spo_main, mq_opt_, norb, nTiles, mq_opt_.splines_per_block, lattice_b, mq_opt_.device_number);
  

  if (!mq_opt_.useRef)
    app_summary() << "Using SoA distance table, Jastrow + einspline, " << '\n'
                  << "and determinant update." << '\n'
		  << "with " << mq_opt_.device_number << " device implementation" << '\n';
  else
    app_summary() << "Using the reference implementation for Jastrow, " << '\n'
                  << "determinant update, and distance table + einspline of the " << '\n'
                  << "reference implementation " << '\n';

  if (mq_opt_.enableCrowd)
    app_summary() << "batched walkers activated \n"
                  << "pack size: " << mq_opt_.pack_size << '\n';
  mq_opt_.Timers[Timer_Init]->stop();

  //Now lets figure out what threading sizes are needed:
  //  For walker level parallelism:
}


void MiniqmcDriver::run()
{
  using MyHandler = decltype(hana::unpack(devices_range, hana::template_<CaseHandler>))::type;

  MyHandler handler(*this);

  handler.run(mq_opt_.device_number);

  results();
}

} // namespace qmcplusplus
