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

#include "Drivers/MiniQMCOptions.h"
#include "Host/OutputManager.h"

namespace qmcplusplus
{
MiniQMCOptions readOptions(int argc, char** argv)
{
  MiniQMCOptions mq_opt;
  using QMCT = QMCTraits;
  int opt;
  while (optind < argc)
  {
    if ((opt = getopt(argc, argv, "bhjvVa:c:C:d:g:m:n:N:r:s:t:w:x:z:")) != -1)
    {
      switch (opt)
      {
      case 'a':
        mq_opt.splines_per_block = atoi(optarg);
        break;
      case 'b':
        mq_opt.useRef = true;
        break;
      case 'C':
	  mq_opt.num_crowds = atoi(optarg);
	  break;
      case 'c':
        mq_opt.crowd_size = atoi(optarg);
        break;
      // case 'd': // Device choice
      //   mq_opt.device_number = atoi(optarg);
      //   break;
      case 'g': // tiling1 tiling2 tiling3
        sscanf(optarg, "%d %d %d", &mq_opt.na, &mq_opt.nb, &mq_opt.nc);
        break;
      case 'h':
        mq_opt.print_help();
        //should throw
        mq_opt.valid = false;
        return mq_opt;
        break;
      case 'j':
        mq_opt.enableJ3 = true;
        break;
      case 'm':
      {
        const QMCT::RealType meshfactor = atof(optarg);
        mq_opt.nx *= meshfactor;
        mq_opt.ny *= meshfactor;
        mq_opt.nz *= meshfactor;
      }
      break;
      case 'n':
        mq_opt.nsteps = atoi(optarg);
        break;
      case 'N':
        mq_opt.nsubsteps = atoi(optarg);
        break;
      case 'r':
        mq_opt.accept_ratio = atof(optarg);
        break;
      case 's':
        mq_opt.iseed = atoi(optarg);
        break;
      case 't':
        mq_opt.timer_level_name = std::string(optarg);
        break;
      case 'v':
        mq_opt.verbose = true;
        break;
      case 'V':
        ::print_version(true);
        return mq_opt;
        break;
      case 'w':
        mq_opt.walkers_per_rank = atoi(optarg);
        break;
      case 'x':
        mq_opt.Rmax = atof(optarg);
        break;
      default:
        break;
      }
    }
    else // disallow non-option arguments
    {
      app_error() << "Non-option arguments not allowed = " << std::endl;
      mq_opt.valid = false;
      mq_opt.print_help();
      return mq_opt;
    }
  }
  return mq_opt;
}

void MiniQMCOptions::print_help()
{
  app_summary() << "usage:" << '\n';
  app_summary() << "  miniqmc   [-bhjvV] [-g \"n0 n1 n2\"] [-m meshfactor]" << '\n';
  app_summary() << "            [-n steps] [-N substeps] [-x rmax]" << '\n';
  app_summary() << "            [-r AcceptanceRatio] [-s seed] [-w walkers]" << '\n';
  app_summary() << "            [-a tile_size] [-t timer_level]" << '\n';
  app_summary() << "options:" << '\n';
  app_summary() << "  -a  splines per spline block       default: num of orbs" << '\n';
  app_summary() << "  -b  use reference implementations  default: off" << '\n';
  app_summary() << "  -g  set the 3D tiling.             default: 1 1 1" << '\n';
  app_summary() << "  -h  print help and exit" << '\n';
  app_summary() << "  -j  enable three body Jastrow      default: off" << '\n';
  app_summary() << "  -m  meshfactor                     default: 1.0" << '\n';
  app_summary() << "  -n  number of MC steps             default: 5" << '\n';
  app_summary() << "  -N  number of MC substeps          default: 1" << '\n';
  app_summary() << "  -c  crowd size                     default: 1\n";
  app_summary() << "  -C  number of crowds               default: 1\n";
  app_summary() << "  -r  set the acceptance ratio.      default: 0.5" << '\n';
  app_summary() << "  -s  set the random seed.           default: 11" << '\n';
  app_summary() << "  -t  timer level: coarse or fine    default: fine" << '\n';
  app_summary() << "  -w  walkers per rank               default: 1" << '\n';
  app_summary() << "  -v  verbose output" << '\n';
  app_summary() << "  -V  print version information and exit" << '\n';
  app_summary() << "  -x  set the Rmax.                  default: 1.7" << '\n';
  //app_summary() << "  -d  device implementation.         default: CPU          " << '\n';
  //app_summary() << "      Available devices:" << '\n';
  // hana::for_each(devices_range, [&](auto x) {
  // 				    std::string enum_name(hana::to<char const*>(device_names[x]));
  // 				    app_summary() << "                         " << x << ".  " << enum_name << '\n';
  // 				});
}

} // namespace qmcplusplus
