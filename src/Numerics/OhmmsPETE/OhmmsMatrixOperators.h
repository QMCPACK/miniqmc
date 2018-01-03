////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by:
// Jeremy McMinnis, jmcminis@gmail.com,
//    University of Illinois at Urbana-Champaign
//
// File created by:
// Jeongnim Kim, jeongnim.kim@gmail.com,
//    University of Illinois at Urbana-Champaign
////////////////////////////////////////////////////////////////////////////////

#ifndef OHMMS_MATRIXOPERATOR_H
#define OHMMS_MATRIXOPERATOR_H


namespace qmcplusplus
{

template <class T1, class C1, class RHS>
inline Matrix<T1, C1> &assign(Matrix<T1, C1> &lhs, const RHS &rhs)
{
  typedef typename CreateLeaf<RHS>::Leaf_t Leaf_t;
  evaluate(lhs, OpAssign(),
           MakeReturn<Leaf_t>::make(CreateLeaf<RHS>::make(rhs)));
  return lhs;
}

template <class T1, class C1, class RHS>
inline Matrix<T1, C1> &operator+=(Matrix<T1, C1> &lhs, const RHS &rhs)
{
  typedef typename CreateLeaf<RHS>::Leaf_t Leaf_t;
  evaluate(lhs, OpAddAssign(),
           MakeReturn<Leaf_t>::make(CreateLeaf<RHS>::make(rhs)));
  return lhs;
}

template <class T1, class C1, class RHS>
inline Matrix<T1, C1> &operator-=(Matrix<T1, C1> &lhs, const RHS &rhs)
{
  typedef typename CreateLeaf<RHS>::Leaf_t Leaf_t;
  evaluate(lhs, OpSubtractAssign(),
           MakeReturn<Leaf_t>::make(CreateLeaf<RHS>::make(rhs)));
  return lhs;
}

template <class T1, class C1, class RHS>
inline Matrix<T1, C1> &operator*=(Matrix<T1, C1> &lhs, const RHS &rhs)
{
  typedef typename CreateLeaf<RHS>::Leaf_t Leaf_t;
  evaluate(lhs, OpMultiplyAssign(),
           MakeReturn<Leaf_t>::make(CreateLeaf<RHS>::make(rhs)));
  return lhs;
}

}

#endif // OHMMS_MATRIXOPERATOR_H

