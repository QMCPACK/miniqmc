//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by: Miguel Morales, moralessilva2@llnl.gov, Lawrence Livermore National Laboratory
//                    Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//                    Jeremy McMinnis, jmcminis@gmail.com, University of Illinois at Urbana-Champaign
//                    Jaron T. Krogel, krogeljt@ornl.gov, Oak Ridge National Laboratory
//                    Mark A. Berrill, berrillma@ornl.gov, Oak Ridge National Laboratory
//
// File created by: Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//////////////////////////////////////////////////////////////////////////////////////


#ifndef QMCPLUSPLUS_SCLAR_TRAITS_H
#define QMCPLUSPLUS_SCLAR_TRAITS_H

#include <complex>

namespace qmcplusplus
{

template<class T>
struct scalar_traits
{
  enum
  {
    DIM = 1
  };
  typedef T real_type;
  typedef T value_type;
  static inline T* get_address(T* a) { return a; }
};

template<typename T>
struct scalar_traits<std::complex<T>>
{
  enum
  {
    DIM = 2
  };
  typedef T real_type;
  typedef std::complex<T> value_type;
  static inline T* get_address(std::complex<T>* a) { return reinterpret_cast<T*>(a); }
};

} // namespace qmcplusplus
#endif
