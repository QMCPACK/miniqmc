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

#ifndef OHMMS_TENSOR_OPERATORS_H
#define OHMMS_TENSOR_OPERATORS_H

#include <Kokkos_Core.hpp>

/*** Tenor operators.  Generic operators are specialized for 1,2 and 3 D
 */
namespace qmcplusplus
{
template<class T1, class T2, class OP, unsigned D>
struct OTAssign<Tensor<T1, D>, Tensor<T2, D>, OP>
{
  KOKKOS_INLINE_FUNCTION static void apply(Tensor<T1, D>& lhs, const Tensor<T2, D>& rhs, OP op)
  {
    for (unsigned d = 0; d < D * D; ++d)
      op(lhs[d], rhs[d]);
  }
};

template<class T1, class T2, class OP, unsigned D>
struct OTAssign<Tensor<T1, D>, T2, OP>
{
  KOKKOS_INLINE_FUNCTION static void apply(Tensor<T1, D>& lhs, T2 rhs, OP op)
  {
    for (unsigned d = 0; d < D * D; ++d)
      op(lhs[d], rhs);
  }
};


//////////////////////////////////////////////////////////////////////
//
// Specializations for Tensors of arbitrary size.
//
//////////////////////////////////////////////////////////////////////

template<class T1, class T2, class OP, unsigned D>
struct OTBinary<Tensor<T1, D>, Tensor<T2, D>, OP>
{
  typedef typename BinaryReturn<T1, T2, OP>::Type_t Type_t;
  KOKKOS_INLINE_FUNCTION static Tensor<Type_t, D>
      apply(const Tensor<T1, D>& lhs, const Tensor<T2, D>& rhs, OP op)
  {
    Tensor<Type_t, D> ret;
    for (unsigned d = 0; d < D * D; ++d)
      ret[d] = op(lhs[d], rhs[d]);
    return ret;
  }
};

template<class T1, class T2, class OP, unsigned D>
struct OTBinary<Tensor<T1, D>, T2, OP>
{
  typedef typename BinaryReturn<T1, T2, OP>::Type_t Type_t;
  KOKKOS_INLINE_FUNCTION static Tensor<Type_t, D> apply(const Tensor<T1, D>& lhs, T2 rhs, OP op)
  {
    Tensor<Type_t, D> ret;
    for (unsigned d = 0; d < D * D; ++d)
      ret[d] = op(lhs[d], rhs);
    return ret;
  }
};

template<class T1, class T2, class OP, unsigned D>
struct OTBinary<T1, Tensor<T2, D>, OP>
{
  typedef typename BinaryReturn<T1, T2, OP>::Type_t Type_t;
  KOKKOS_INLINE_FUNCTION static Tensor<Type_t, D> apply(T1 lhs, const Tensor<T2, D>& rhs, OP op)
  {
    Tensor<Type_t, D> ret;
    for (unsigned d = 0; d < D * D; ++d)
      ret[d] = op(lhs, rhs[d]);
    return ret;
  }
};

//////////////////////////////////////////////////////
//
// determinant: generalized
//
//////////////////////////////////////////////////////
template<class T, unsigned D>
KOKKOS_INLINE_FUNCTION typename Tensor<T, D>::Type_t det(const Tensor<T, D>& a)
{
  // to implement the general case here
  return 0;
}

//////////////////////////////////////////////////////
// specialized for D=1
//////////////////////////////////////////////////////
template<class T>
KOKKOS_INLINE_FUNCTION typename Tensor<T, 1>::Type_t det(const Tensor<T, 1>& a)
{
  return a(0, 0);
}

//////////////////////////////////////////////////////
// specialized for D=2
//////////////////////////////////////////////////////
template<class T>
KOKKOS_INLINE_FUNCTION typename Tensor<T, 2>::Type_t det(const Tensor<T, 2>& a)
{
  return a(0, 0) * a(1, 1) - a(0, 1) * a(1, 0);
}

//////////////////////////////////////////////////////
// specialized for D=3
//////////////////////////////////////////////////////
template<class T>
KOKKOS_INLINE_FUNCTION typename Tensor<T, 3>::Type_t det(const Tensor<T, 3>& a)
{
  return a(0, 0) * (a(1, 1) * a(2, 2) - a(1, 2) * a(2, 1)) +
      a(0, 1) * (a(1, 2) * a(2, 0) - a(1, 0) * a(2, 2)) +
      a(0, 2) * (a(1, 0) * a(2, 1) - a(1, 1) * a(2, 0));
}

//////////////////////////////////////////////////////
//
// inverse: generalized
// A*B = I, * being the matrix multiplication  I(i,j) = sum_k A(i,k)*B(k,j)
//
//////////////////////////////////////////////////////
template<class T, unsigned D>
KOKKOS_INLINE_FUNCTION Tensor<T, D> inverse(const Tensor<T, D>& a)
{
  return Tensor<T, D>();
}

//////////////////////////////////////////////////////
// specialized for D=1
//////////////////////////////////////////////////////
template<class T>
KOKKOS_INLINE_FUNCTION Tensor<T, 1> inverse(const Tensor<T, 1>& a)
{
  return Tensor<T, 1>(1.0 / a(0, 0));
}

//////////////////////////////////////////////////////
// specialized for D=2
//////////////////////////////////////////////////////
template<class T>
KOKKOS_INLINE_FUNCTION Tensor<T, 2> inverse(const Tensor<T, 2>& a)
{
  T vinv = 1 / det(a);
  return Tensor<T, 2>(vinv * a(1, 1), -vinv * a(0, 1), -vinv * a(1, 0), vinv * a(0, 0));
}

//////////////////////////////////////////////////////
// specialized for D=3
//////////////////////////////////////////////////////
template<class T>
KOKKOS_INLINE_FUNCTION Tensor<T, 3> inverse(const Tensor<T, 3>& a)
{
  T vinv = 1 / det(a);
  return Tensor<T, 3>(vinv * (a(1, 1) * a(2, 2) - a(1, 2) * a(2, 1)),
                      vinv * (a(2, 1) * a(0, 2) - a(2, 2) * a(0, 1)),
                      vinv * (a(0, 1) * a(1, 2) - a(0, 2) * a(1, 1)),
                      vinv * (a(1, 2) * a(2, 0) - a(1, 0) * a(2, 2)),
                      vinv * (a(2, 2) * a(0, 0) - a(2, 0) * a(0, 2)),
                      vinv * (a(0, 2) * a(1, 0) - a(0, 0) * a(1, 2)),
                      vinv * (a(1, 0) * a(2, 1) - a(1, 1) * a(2, 0)),
                      vinv * (a(2, 0) * a(0, 1) - a(2, 1) * a(0, 0)),
                      vinv * (a(0, 0) * a(1, 1) - a(0, 1) * a(1, 0)));
}

//////////////////////////////////////////////////////////////////////
//
// Specializations for Tensor dot Tensor
//
//////////////////////////////////////////////////////////////////////

template<class T1, class T2, unsigned D>
struct OTDot<Tensor<T1, D>, Tensor<T2, D>>
{
  typedef typename BinaryReturn<T1, T2, OpMultiply>::Type_t Type_t;
  KOKKOS_INLINE_FUNCTION static Tensor<Type_t, D>
      apply(const Tensor<T1, D>& lhs, const Tensor<T2, D>& rhs)
  {
    Tensor<Type_t, D> res = Tensor<Type_t, D>::DontInitialize();
    for (unsigned int i = 0; i < D; ++i)
      for (unsigned int j = 0; j < D; ++j)
      {
        Type_t sum = lhs(i, 0) * rhs(0, j);
        for (unsigned int k = 1; k < D; ++k)
          sum += lhs(i, k) * rhs(k, j);
        res(i, j) = sum;
      }
    return res;
  }
};

template<class T1, class T2>
struct OTDot<Tensor<T1, 1>, Tensor<T2, 1>>
{
  typedef typename BinaryReturn<T1, T2, OpMultiply>::Type_t Type_t;
  KOKKOS_INLINE_FUNCTION static Tensor<Type_t, 1>
      apply(const Tensor<T1, 1>& lhs, const Tensor<T2, 1>& rhs)
  {
    return Tensor<Type_t, 1>(lhs[0] * rhs[0]);
  }
};

template<class T1, class T2>
struct OTDot<Tensor<T1, 2>, Tensor<T2, 2>>
{
  typedef typename BinaryReturn<T1, T2, OpMultiply>::Type_t Type_t;
  KOKKOS_INLINE_FUNCTION static Tensor<Type_t, 2>
      apply(const Tensor<T1, 2>& lhs, const Tensor<T2, 2>& rhs)
  {
    return Tensor<Type_t, 2>(lhs(0, 0) * rhs(0, 0) + lhs(0, 1) * rhs(1, 0),
                             lhs(0, 0) * rhs(0, 1) + lhs(0, 1) * rhs(1, 1),
                             lhs(1, 0) * rhs(0, 0) + lhs(1, 1) * rhs(1, 0),
                             lhs(1, 0) * rhs(0, 1) + lhs(1, 1) * rhs(1, 1));
  }
};

template<class T1, class T2>
struct OTDot<Tensor<T1, 3>, Tensor<T2, 3>>
{
  typedef typename BinaryReturn<T1, T2, OpMultiply>::Type_t Type_t;
  KOKKOS_INLINE_FUNCTION static Tensor<Type_t, 3>
      apply(const Tensor<T1, 3>& lhs, const Tensor<T2, 3>& rhs)
  {
    return Tensor<Type_t, 3>(lhs(0, 0) * rhs(0, 0) + lhs(0, 1) * rhs(1, 0) + lhs(0, 2) * rhs(2, 0),
                             lhs(0, 0) * rhs(0, 1) + lhs(0, 1) * rhs(1, 1) + lhs(0, 2) * rhs(2, 1),
                             lhs(0, 0) * rhs(0, 2) + lhs(0, 1) * rhs(1, 2) + lhs(0, 2) * rhs(2, 2),
                             lhs(1, 0) * rhs(0, 0) + lhs(1, 1) * rhs(1, 0) + lhs(1, 2) * rhs(2, 0),
                             lhs(1, 0) * rhs(0, 1) + lhs(1, 1) * rhs(1, 1) + lhs(1, 2) * rhs(2, 1),
                             lhs(1, 0) * rhs(0, 2) + lhs(1, 1) * rhs(1, 2) + lhs(1, 2) * rhs(2, 2),
                             lhs(2, 0) * rhs(0, 0) + lhs(2, 1) * rhs(1, 0) + lhs(2, 2) * rhs(2, 0),
                             lhs(2, 0) * rhs(0, 1) + lhs(2, 1) * rhs(1, 1) + lhs(2, 2) * rhs(2, 1),
                             lhs(2, 0) * rhs(0, 2) + lhs(2, 1) * rhs(1, 2) + lhs(2, 2) * rhs(2, 2));
  }
};
} // namespace qmcplusplus

#endif // OHMMS_TENSOR_OPERATORS_H
