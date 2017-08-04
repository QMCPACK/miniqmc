////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by: 
// Jeremy McMinnis, jmcminis@gmail.com,
//    University of Illinois at Urbana-Champaign
// Jeongnim Kim, jeongnim.kim@gmail.com,
//    University of Illinois at Urbana-Champaign
//
// File created by: 
// Jeongnim Kim, jeongnim.kim@gmail.com,
//    University of Illinois at Urbana-Champaign
////////////////////////////////////////////////////////////////////////////////

#ifndef OHMMS_LATTICEOPERATIONS_H
#define OHMMS_LATTICEOPERATIONS_H
#include "OhmmsPETE/TinyVector.h"

namespace qmcplusplus
{

template <class T, unsigned D> struct CheckBoxConds
{
  inline static bool inside(const TinyVector<T, D> &u)
  {
    bool yes = (u[0] > 0.0 && u[0] < 1.0);
    for (int i = 1; i < D; ++i) yes &= (u[i] > 0.0 && u[i] < 1.0);
    return yes;
  }

  inline static bool inside(const TinyVector<T, D> &u, TinyVector<T, D> &ubox)
  {
    for (int i = 0; i < D; ++i) ubox[i] = u[i] - std::floor(u[i]);
    return true;
  }
};

template <class T> struct CheckBoxConds<T, 3>
{
  inline static bool inside(const TinyVector<T, 3> &u)
  {
    return (u[0] > 0.0 && u[0] < 1.0) && (u[1] > 0.0 && u[1] < 1.0) &&
           (u[2] > 0.0 && u[2] < 1.0);
  }

  inline static bool inside(const TinyVector<T, 3> &u,
                            const TinyVector<int, 3> &bc)
  {
    return (bc[0] || (u[0] > 0.0 && u[0] < 1.0)) &&
           (bc[1] || (u[1] > 0.0 && u[1] < 1.0)) &&
           (bc[2] || (u[2] > 0.0 && u[2] < 1.0));
  }

  inline static bool inside(const TinyVector<T, 3> &u, TinyVector<T, 3> &ubox)
  {
    ubox[0] = u[0] - std::floor(u[0]);
    ubox[1] = u[1] - std::floor(u[1]);
    ubox[2] = u[2] - std::floor(u[2]);
    return true;
  }
};
}

#endif
