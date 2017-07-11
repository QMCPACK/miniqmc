#include <iostream>
#include <mpi/detail/communicator.hpp>

#if defined(HAVE_MPI)
namespace qmcplusplus { namespace mpi {

    ///forward declaration 
    MPI_Comm initialize(int argc, char** argv);

    environment::environment(int argc, char** argv)
    {
      initialize(argc,argv);
    }

    environment::~environment()
    {
      int flag;
      int success = MPI_Initialized(&flag);
      if(flag)
        MPI_Finalize();
    }

    communicator::communicator():impl_(MPI_COMM_NULL)
    {
      impl_=initialize(0,nullptr);
      MPI_Comm_size(impl_, &num_ranks);
      MPI_Comm_rank(impl_, &my_rank);
    }

    communicator::communicator(const communicator& comm)
    {
      int success=MPI_Comm_dup(comm.impl_,&impl_);
      MPI_Comm_size(impl_, &num_ranks);
      MPI_Comm_rank(impl_, &my_rank);
    }

    communicator::~communicator()
    {
      if(impl_!=MPI_COMM_WORLD)
        MPI_Comm_free(&impl_);
      impl_=MPI_COMM_NULL;
    }

    communicator communicator::split(int color) const
    {
      communicator newcomm;
      MPI_Comm_split(impl_,color,my_rank,&(newcomm.impl_));
      MPI_Comm_size(newcomm, &(newcomm.num_ranks));
      MPI_Comm_rank(newcomm, &(newcomm.my_rank));
      return newcomm;
    }

    MPI_Comm initialize(int argc, char** argv)
    {
      int flag;
      int success = MPI_Initialized(&flag);
      if(flag == 0)
      {
#if defined(_OPENMP)
        int provided, claimed;
        success=MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
        MPI_Query_thread(&claimed);
        if (claimed != provided) 
        {
          MPI_Abort(MPI_COMM_WORLD,0);
        }
#else
	success = MPI_Init(&argc,&argv);
#endif
      }
      return MPI_COMM_WORLD;
    }

#if 0
    communicator communicator::split(int color, int key) const
    {
      communicator newcomm;
      MPI_Comm_split(impl_,color,key,&(newcomm.impl_));
      MPI_Comm_size(newcomm, &(newcomm.num_ranks));
      MPI_Comm_rank(newcomm, &(newcomm.my_rank));
      return newcomm;
    }
#endif
}} 
#else
namespace qmcplusplus { namespace mpi {

  environment::environment(int argc, char** argv) { } 
  environment::~environment() { }

  communicator::communicator():num_ranks(1),my_rank(0) { }

  communicator::communicator(const communicator& comm):num_ranks(1),my_rank(0) { }

  communicator::~communicator() { }

  communicator communicator::split(int color) const
  {
    return communicator();
  }
}}
#endif

