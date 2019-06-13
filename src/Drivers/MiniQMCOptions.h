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

#ifndef QMCPLUSPLUS_MINIQMC_OPTIONS_H
#define QMCPLUSPLUS_MINIQMC_OPTIONS_H

#include <getopt.h>
#include <iostream>
#include <cstdio>
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
  Timer_evalVGH,
  Timer_ratioGrad,
  Timer_Update,
};

/** Reads and holds the options to support more flexible non 'c main()' drivers
 *  
 */
class MiniQMCOptions
{
public:
  TimerNameList_t<MiniQMCTimers> MiniQMCTimerNames{
      {Timer_Total, "Total"},
      {Timer_Init, "Initialization"},
      {Timer_Diffusion, "Diffusion"},
      {Timer_ECP, "Pseudopotential"},
      {Timer_Value, "Value"},
      {Timer_evalGrad, "Current Gradient"},
      {Timer_evalVGH, "Spline Hessian Evaluation"},
      {Timer_ratioGrad, "New Gradient"},
      {Timer_Update, "Update"},
  };

  static void print_help();

  using QMCT        = QMCTraits;
  //Devices device    = Devices::CPU;
  bool valid        = true;
  //int device_number = 0;
  int na            = 1;
  int nb            = 1;
  int nc            = 1;
  int nsteps        = 5;
  int iseed         = 11;
  int nx = 37, ny = 37, nz = 37;
  int num_crowds = 1;
  int splines_per_block = -1;
  int nsubsteps         = 1;
  int nels = 0;
  // Set cutoff for NLPP use.
  // This makes precision an issue to select at run time.
  QMCT::RealType Rmax          = 1.7;
  QMCT::RealType accept_ratio  = 0.5;
  //useRef is a particular implementation of numerous objects fix that remove this option
  //and many branch statements
  bool useRef                  = false;
  bool enableJ3                = false;
  bool enableCrowd             = false;
  bool verbose                 = false;
  std::string timer_level_name = "fine";
  TimerList_t Timers;
  int crowd_size  = 1;
  int walkers_per_rank = 0;
  MiniQMCOptions()                      = default;
  MiniQMCOptions(const MiniQMCOptions&) = default;
};

MiniQMCOptions readOptions(int argc,char** argv);

} // namespace qmcplusplus

#endif
