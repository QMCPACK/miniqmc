#include "MiniqmcOptions.hpp"
namespace qmcplusplus
{
MiniqmcOptions readOptions(int argc, char** argv)
{
  MiniqmcOptions mq_opt;
  using QMCT = QMCTraits;
  int opt;
  while (optind < argc)
  {
    if ((opt = getopt(argc, argv, "bhjvVMa:c:d:g:m:n:p:N:r:s:t:w:x:z:")) != -1)
    {
      switch (opt)
      {
      case 'a':
        mq_opt.tileSize = atoi(optarg);
        break;
      case 'b':
        mq_opt.useRef = true;
        break;
      case 'p': // pack_size
        mq_opt.pack_size = atoi(optarg);
	app_error() << "Read pack_size = " << optarg << '\n';
        break;
      case 'c': // number of members per team
        mq_opt.team_size = atoi(optarg);
        break;
      case 'd': // Device choice
        mq_opt.device_number = atoi(optarg);
        break;
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
      case 'M':
	mq_opt.enableMovers = true;
	break;
      case 'n':
        mq_opt.nsteps = atoi(optarg);
        break;
      case 'N':
        mq_opt.nsubsteps = atoi(optarg);
        break;
      case 'r': // accept
        mq_opt.accept = atof(optarg);
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
        //should throw
        break;
      case 'w': // number of nmovers
        mq_opt.nmovers = atoi(optarg);
        break;
      case 'x': // rmax
        mq_opt.Rmax = atof(optarg);
        break;
      case 'z': //number of crews
        mq_opt.ncrews = atoi(optarg);
        break;
      default: // kokkos arguments may be passed
        break;
      }
    }
    else // disallow non-option arguments
    {
      app_error() << "Non-option arguments not allowed = " <<   std::endl;
      mq_opt.valid = false;
      mq_opt.print_help();
      return mq_opt;
      //should throw
    }
  }
  return mq_opt;
}
} // namespace qmcplusplus
