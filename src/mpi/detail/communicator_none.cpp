#include <iostream>
#include <mpi/detail/communicator.hpp>

namespace qmcplusplus {
  namespace mpi {

    environment::environment(int argc, char** argv) { } 
    environment::~environment() { }

    communicator::communicator():num_ranks(1),my_rank(0) { }

    communicator::communicator(const communicator& comm):num_ranks(1),my_rank(0) { }

    communicator::~communicator() { }

    communicator communicator::split(int color) const
    {
      return communicator();
    }
  }
}
