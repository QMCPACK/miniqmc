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
    
    
/** @file collectives.hpp
 *
 * Minimal MPI collectives
 * - Only reduce, allreduce, bcast for numerical types are supported.
 * - Reduction supports only sum, max and min
 * - No warning and error handling.
 */

#ifndef QMCPLUSPLUS_COLLECTIVES_H
#define QMCPLUSPLUS_COLLECTIVES_H
#include <mpi/detail/communicator.hpp>
namespace qmcplusplus { namespace mpi {

  template<typename T>
    struct maximum: public std::binary_function<T, T, T>
    {
       const T& operator()(const T& x, const T& y) const
       {
          return x < y? y : x;
       }
    };
 
  template<typename T>
    struct minimum: public std::binary_function<T, T, T>
    {
       const T& operator()(const T& x, const T& y) const
       {
          return x > y? y : x;
       }
    };

#if defined(HAVE_MPI)
  //a simplified implementation of is_mpi_op
  template<typename Op, typename T> struct cast_mpi_op { };
  
  template<typename T>
    struct cast_mpi_op<std::plus<T>,T>
    {
      static MPI_Op op() { return MPI_SUM; }
    };
 
  template<typename T>
    struct cast_mpi_op<maximum<T>,T>
    {
      static MPI_Op op() { return MPI_MAX; }
    };
 
  template<typename T>
    struct cast_mpi_op<minimum<T>,T>
    {
      static MPI_Op op() { return MPI_MIN; }
    };
 
  template<typename T, typename Op> 
    void all_reduce(const communicator& comm, const T* value, int n, T* out_value, Op )
    { 
      MPI_Allreduce(value, out_value, n, get_mpi_datatype(value[0]),cast_mpi_op<Op,T>::op(),comm);
    }
 
  template<typename T, typename Op> 
    void all_reduce(const communicator& comm, const T& value, T& out_value, Op op)
    {
      MPI_Allreduce(&value, &out_value, 1, get_mpi_datatype(value),cast_mpi_op<Op,T>::op(),comm);
    }
 
  template<typename T, typename Op>
    void reduce(const communicator& comm, const T& in_value, T& out_value, Op op, int root)
    {
      MPI_Reduce(&in_value, &out_value, 1, get_mpi_datatype(in_value),cast_mpi_op<Op,T>::op(),root,comm);
    }
 
  template<typename T, typename Op>
    void reduce(const communicator& comm, const T* in_values, int n, T* out_values, Op op, int root)
    {
      MPI_Reduce(in_values, out_values, n, get_mpi_datatype(in_values[0]),cast_mpi_op<Op,T>::op(),root,comm);
    }
 
   template<typename T>
    void broadcast(const communicator& comm, T& value, int root)
    {
      MPI_Bcast(&value,1,get_mpi_datatype(value),root,comm);
    }
 
  template<typename T>
    void broadcast(const communicator& comm, T* values, int n, int root)
    {
       MPI_Bcast(values,n,get_mpi_datatype(values[0]),root,comm);
    }
#else
  template<typename T, typename Op> 
    void all_reduce(const communicator& comm, const T* value, int n, T* out_value, Op )
    {
      std::copy(value,value+n,out_value);
    }
 
  template<typename T, typename Op> 
    void all_reduce(const communicator& comm, const T& value, T& out_value, Op op)
    {
      out_value=value;
    }
 
  template<typename T, typename Op>
    void reduce(const communicator& comm, const T& in_value, T& out_value, Op op, int root) 
    {
      out_value=in_value;
    }
 
  template<typename T, typename Op>
    void reduce(const communicator& comm, const T* in_values, int n, T* out_values, Op op, int root)
    {
      std::copy(in_values,in_values+n,out_values);
    }
 
   template<typename T>
    void broadcast(const communicator& comm, T& value, int root){}
 
  template<typename T>
    void broadcast(const communicator& comm, T* values, int n, int root){}
#endif
}}
#endif 
