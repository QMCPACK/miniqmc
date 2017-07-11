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
    
#include <complex>

/** @file  datatypes.hpp
 *
 * data type compatible with MPI. Provides minimalistic interfaces to MPI
 * Do not use "using namespace mpi". Always use with mpi::
 * Expect to be replaced by boost::mpi
 */
#ifndef TEMP_MPI_DATATYPEDEFINE_H
#define TEMP_MPI_DATATYPEDEFINE_H

namespace qmcplusplus {
namespace mpi {

#if defined(HAVE_MPI)
template <typename T>
inline MPI_Datatype
get_mpi_datatype(const T&)
{
  return MPI_BYTE;
}

//specialization for MPI data type
//
#define TEMP_MPI_DATATYPE(CppType, MPITYPE)              \
template<>                                               \
inline MPI_Datatype                                      \
get_mpi_datatype< CppType >(const CppType&) { return MPITYPE; }

TEMP_MPI_DATATYPE(short, MPI_SHORT);

TEMP_MPI_DATATYPE(int, MPI_INT);

TEMP_MPI_DATATYPE(long, MPI_LONG);

TEMP_MPI_DATATYPE(float, MPI_FLOAT);

TEMP_MPI_DATATYPE(double, MPI_DOUBLE);

TEMP_MPI_DATATYPE(long double, MPI_LONG_DOUBLE);

TEMP_MPI_DATATYPE(unsigned char, MPI_UNSIGNED_CHAR);

TEMP_MPI_DATATYPE(unsigned short, MPI_UNSIGNED_SHORT);

TEMP_MPI_DATATYPE(unsigned int, MPI_UNSIGNED);

TEMP_MPI_DATATYPE(unsigned long, MPI_UNSIGNED_LONG);

TEMP_MPI_DATATYPE(std::complex<double>, MPI_COMPLEX);

TEMP_MPI_DATATYPE(std::complex<float>, MPI_DOUBLE_COMPLEX)

#else

template <typename T>
inline int get_mpi_datatype(const T&) { return 0;}

#endif
}
}//end of mpi
#endif
