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

#ifndef QMCPLUSPLUS_MPI_MASTER_H
#define QMCPLUSPLUS_MPI_MASTER_H

#include <mpi/detail/communicator.hpp>
#include <mpi/detail/collectives.hpp>

namespace qmcplusplus { namespace mpi {

  /** addition reduction functions specialized with Op==MPI_SUM */
  template<typename T>
   void all_reduce(const communicator& comm, const T* value, int n, T* out_value)
   { 
     all_reduce(comm,value,n,out_value,std::plus<T>());
   }

  template<typename T>
    void all_reduce(const communicator& comm, const T& value, T& out_value)
    {
      all_reduce(comm,value,out_value,std::plus<T>());
    }

  template<typename T>
    void reduce(const communicator& comm, const T& in_value, T& out_value, int root)
    {
      reduce(comm,in_value,out_value,root,std::plus<T>(),root);
    }
 
  template<typename T>
    void reduce(const communicator& comm, const T* in_values, int n, T* out_values, int root)
    {
      reduce(comm,in_values,n,out_values,std::plus<T>(),root);
    }
}}
#endif
