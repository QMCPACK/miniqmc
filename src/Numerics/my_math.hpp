////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2019 QMCPACK developers.
//
// File developed by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//
// File created by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
////////////////////////////////////////////////////////////////////////////////

#ifndef QMCPLUSPLUS_MY_MATH_HPP
#define QMCPLUSPLUS_MY_MATH_HPP

namespace qmcplusplus
{
  template<typename T,
    typename = typename std::enable_if<std::is_floating_point<T>::value>::type>
  T my_modf(T x, T* intpart)
  {
    T dx;
    if (x < 0)
    {
      *intpart = - static_cast<T>(static_cast<int>(-x));
      dx = x - *intpart;
    }
    else
    {
      *intpart = static_cast<T>(static_cast<int>(x));
      dx = x - *intpart;
    }
    return dx;
  }
} // namespace qmcplusplus
#endif
