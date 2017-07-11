//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by: 
//
// File created by: Jeongnim Kim, jeongnim.kim@inte.com, Intel Corp.
//////////////////////////////////////////////////////////////////////////////////////
    
    
/** @file communicator.hpp
 *
 * Minimal MPI implementation compatible to boost.mpi
 * send/recv of primitative types are supported.
 *
 */

#ifndef QMCPLUSPLUS_COMMUNICATOR_H
#define QMCPLUSPLUS_COMMUNICATOR_H
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#if defined(HAVE_MPI)
#include <mpi.h>
#else
typedef int MPI_Comm;
typedef int MPI_Status;
typedef int MPI_Request;
#endif
#include "mpi/detail/datatypes.hpp"
#include "mpi/detail/status.hpp"
#include "mpi/detail/request.hpp"

namespace qmcplusplus { namespace mpi {

  /** Set up MPI and threading environment
   *
   * All executable should include 
   * mpi::environment env(argc,argv);
   */
  struct environment 
  {
    environment()=delete;
    environment(const environment&)=delete;
    environment(int argc, char**argv);
    ~environment();
  };

  class request;

  /** MPI communicator class
   */
  class communicator
  {
    public:

      communicator();

      communicator(const communicator& comm);

      ~communicator();

      inline int size() const { return num_ranks; }
      inline int rank() const { return my_rank; }
      inline operator MPI_Comm() const { return impl_; }

      communicator split(int color) const;

      template<typename T> void send(int dest, int tag, const T& value) const;
      template<typename T> void recv(int source, int tag, T& value) const;

      template<typename T> void send(int dest, int tag, const T* values, int n) const;
      template<typename T> void recv(int source, int tag, T* values, int n) const;

      template<typename T> request isend(int dest, int tag, const T& value) const;
      template<typename T> request irecv(int source, int tag, T& value) const;

      template<typename T> request isend(int dest, int tag, const T* values, int n) const;
      template<typename T> request irecv(int source, int tag, T* values, int n) const;

    private:
      MPI_Comm impl_;
      int my_rank;
      int num_ranks;
  };

#if defined(HAVE_MPI)
  template<typename T>
    void communicator::send(int dest, int tag, const T& value) const
    {
      int ierr=MPI_Send(&value,1,get_mpi_datatype(value),dest,tag,impl_);
    }

  template<typename T>
    void communicator::recv(int source, int tag, T& value) const
    {
      MPI_Status status;
      int ierr=MPI_Recv(&value,1,get_mpi_datatype(value),source,tag,impl_,&status);
    }

  template<typename T>
    void communicator::send(int dest, int tag, const T* values, int n) const
    {
      int ierr=MPI_Send(values,n,get_mpi_datatype(values[0]),dest,tag,impl_);
    }

  template<typename T>
    void communicator::recv(int source, int tag, T* values, int n) const
    {
      MPI_Status status;
      int ierr=MPI_Recv(values,n,get_mpi_datatype(values[0]),source,tag,impl_,&status);
    }

  template<typename T>
    request communicator::isend(int dest, int tag, const T& value) const
    {
      request result;
      int ierr=MPI_Isend(&value,1,get_mpi_datatype(value),dest,tag,impl_,result.my_requests);
      return result;
    }

  template<typename T>
    request communicator::irecv(int source, int tag, T& value) const
    {
      request result;
      int ierr=MPI_Irecv(&value,1,get_mpi_datatype(value),dest,tag,impl_,result.my_requests);
      return result;
    }

  template<typename T>
    request communicator::isend(int dest, int tag, const T* values, int n) const
    {
      request result;
      int ierr=MPI_Isend(values,n,get_mpi_datatype(values[0]),dest,tag,impl_,result.my_requests);
      return result;
    }

  template<typename T>
    request communicator::irecv(int source, int tag, T* values, int n) const
    {
      request result;
      int ierr=MPI_Irecv(values,n,get_mpi_datatype(values[0]),source,tag,impl_,result.my_requests);
      return result;
    }

#else
  template<typename T>
    void communicator::send(int dest, int tag, const T& value) const {}
  template<typename T>
    void communicator::recv(int source, int tag, T& value) const{}
  template<typename T>
    void communicator::send(int dest, int tag, const T* values, int n) const {}
  template<typename T>
    void communicator::recv(int source, int tag, T* values, int n) const {}
  template<typename T>
    request communicator::isend(int dest, int tag, const T& value) const
    {
      return request();
    }

  template<typename T>
    request communicator::irecv(int source, int tag, T& value) const
    {
      request result;
      int ierr=MPI_Irecv(&value,1,get_mpi_datatype(value),dest,tag,impl_,result.my_requests);
      return result;
    }

  template<typename T>
    request communicator::isend(int dest, int tag, const T* values, int n) const
    {
      request result;
      int ierr=MPI_Isend(values,n,get_mpi_datatype(values[0]),dest,tag,impl_,result.my_requests);
      return result;
    }

  template<typename T>
    request communicator::irecv(int source, int tag, T* values, int n) const
    {
      request result;
      int ierr=MPI_Irecv(values,n,get_mpi_datatype(values[0]),source,tag,impl_,result.my_requests);
      return result;
    }
#endif

}} 
#endif // OHMMS_COMMUNICATE_H
