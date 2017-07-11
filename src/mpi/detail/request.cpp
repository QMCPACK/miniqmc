#include <mpi/communicator.hpp>
#include <mpi/request.hpp>
#include <mpi/status.hpp>
namespace qmcplusplus { namespace mpi {

#if defined(HAVE_MPI)
  request::request()
  {
    my_requests[0]= MPI_REQUEST_NULL;
    my_requests[1]= MPI_REQUEST_NULL;
  }

  status request::wait()
  {
    MPI_Status status_;
    if(my_requests[0] != MPI_REQUEST_NULL)
    {
      int success=MPI_Wait(my_requests,&status_);
    }
    status result;
    result.m_status=status_;
    return result;
  }
#endif
}}
