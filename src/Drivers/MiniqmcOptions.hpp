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

#ifndef QMCPLUSPLUS_MINIQMC_OPTIONS_H
#define QMCPLUSPLUS_MINIQMC_OPTIONS_H

#include <getopt.h>
#include <iostream>
#include <cstdio>
#include <boost/hana/for_each.hpp>
#include "Devices.h"
#include "Input/Input.hpp"
#include "Utilities/qmcpack_version.h"
#include "Utilities/NewTimer.h"

namespace qmcplusplus
{
  enum MiniQMCTimers
  {
    Timer_Total,
    Timer_Init,
    Timer_Diffusion,
    Timer_ECP,
    Timer_Value,
    Timer_evalGrad,
    Timer_ratioGrad,
    Timer_Update,
  };

/** reads and holds the options
 *  composed into MiniqmcDriver
 */
class MiniqmcOptions
{
public:
  TimerNameList_t<MiniQMCTimers> MiniQMCTimerNames{
      {Timer_Total, "Total"},
      {Timer_Init, "Initialization"},
      {Timer_Diffusion, "Diffusion"},
      {Timer_ECP, "Pseudopotential"},
      {Timer_Value, "Value"},
      {Timer_evalGrad, "Current Gradient"},
      {Timer_ratioGrad, "New Gradient"},
      {Timer_Update, "Update"},
  };

  static void print_help()
  {
    app_summary() << "usage:" << '\n';
    app_summary() << "  miniqmc   [-bhjvV] [-g \"n0 n1 n2\"] [-m meshfactor]" << '\n';
    app_summary() << "            [-n steps] [-N substeps] [-x rmax]" << '\n';
    app_summary() << "            [-r AcceptanceRatio] [-s seed] [-w walkers]" << '\n';
    app_summary() << "            [-a tile_size] [-t timer_level]" << '\n';
    app_summary() << "options:" << '\n';
    app_summary() << "  -a  size of each spline tile       default: num of orbs" << '\n';
    app_summary() << "  -b  use reference implementations  default: off" << '\n';
    app_summary() << "  -g  set the 3D tiling.             default: 1 1 1" << '\n';
    app_summary() << "  -h  print help and exit" << '\n';
    app_summary() << "  -j  enable three body Jastrow      default: off" << '\n';
    app_summary() << "  -m  meshfactor                     default: 1.0" << '\n';
    app_summary() << "  -n  number of MC steps             default: 5" << '\n';
    app_summary() << "  -N  number of MC substeps          default: 1" << '\n';
    app_summary() << "  -r  set the acceptance ratio.      default: 0.5" << '\n';
    app_summary() << "  -s  set the random seed.           default: 11" << '\n';
    app_summary() << "  -t  timer level: coarse or fine    default: fine" << '\n';
    app_summary() << "  -w  number of walker(movers)       default: 1" << '\n';
    app_summary() << "  -v  verbose output" << '\n';
    app_summary() << "  -V  print version information and exit" << '\n';
    app_summary() << "  -x  set the Rmax.                  default: 1.7" << '\n';
    app_summary() << "  -z  number of crews for walker partitioning.   default: 1" << '\n';
    app_summary() << "  -D  device implementation.         default: CPU          " << '\n';
    app_summary() << "      Available devices:" << '\n';
    hana::for_each(devices_range, [&](auto x) {
      std::string enum_name = device_names[x];
      app_summary() << "                         " << x << ".  " << enum_name << '\n';
    });
  }

  using QMCT        = QMCTraits;
  Devices device    = Devices::CPU;
  int device_number = 0;
  int na            = 1;
  int nb            = 1;
  int nc            = 1;
  int nsteps        = 5;
  int iseed         = 11;
  int nx = 37, ny = 37, nz = 37;
  int nmovers = 1;
  // thread blocking
  int tileSize  = -1;
  int team_size = 1;
  int nsubsteps = 1;

  int ncrews    = 1;
  // Set cutoff for NLPP use.
  QMCT::RealType Rmax          = 1.7;
  bool useRef                  = false;
  bool enableJ3                = false;
  bool verbose                 = false;
  std::string timer_level_name = "fine";
  TimerList_t Timers;

  int pack_size = 1;
  
  MiniqmcOptions() = default;
  MiniqmcOptions(const MiniqmcOptions&) = default; // { std::cout << "MiniqmcOptions copy made" << '\n'; }

};

MiniqmcOptions readOptions(int argc, char** argv);

} // namespace qmcplusplus

#endif
