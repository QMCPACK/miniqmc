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

#ifndef OHMMS_TINYVECTOR_OPERATORS_H
#define OHMMS_TINYVECTOR_OPERATORS_H
#include <complex>
#include "clean_inlining.h"

namespace qmcplusplus
{
template<class T1>
struct BinaryReturn<T1, std::complex<T1>, OpMultiply>
{
  typedef std::complex<T1> Type_t;
};

template<class T1>
struct BinaryReturn<std::complex<T1>, T1, OpMultiply>
{
  typedef std::complex<T1> Type_t;
};

///////////////////////////////////////////////////////////////////////
//
// Assignment operators
// template<class T1, class T2, class OP> struct OTAssign {};
//
///////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////
// Specializations for TinyVectors of arbitrary size.
//////////////////////////////////////////////////////////////////////
template<class T1, class T2, class OP, unsigned D>
struct OTAssign<TinyVector<T1, D>, TinyVector<T2, D>, OP>
{
  KOKKOS_INLINE_FUNCTION static void
  apply(TinyVector<T1, D>& lhs, const TinyVector<T2, D>& rhs, OP op)
  {
    for (unsigned d = 0; d < D; ++d)
      op(lhs[d], rhs[d]);
  }
};

template<class T1, class T2, class OP, unsigned D>
struct OTAssign<TinyVector<T1, D>, T2, OP>
{
  KOKKOS_INLINE_FUNCTION static void apply(TinyVector<T1, D>& lhs, const T2& rhs, OP op)
  {
    for (unsigned d = 0; d < D; ++d)
      op(lhs[d], rhs);
  }
};


///////////////////////////////////////////////////////////////////////
//
// Binary operators
//template<class T1, class T2, class OP> struct OTBinary {};
//
///////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////
// Specializations for TinyVectors of arbitrary size.
//////////////////////////////////////////////////////////////////////

template<class T1, class T2, class OP, unsigned D>
struct OTBinary<TinyVector<T1, D>, TinyVector<T2, D>, OP>
{
  typedef typename BinaryReturn<T1, T2, OP>::Type_t Type_t;
  KOKKOS_INLINE_FUNCTION static TinyVector<Type_t, D>
  apply(const TinyVector<T1, D>& lhs, const TinyVector<T2, D>& rhs, OP op)
  {
    TinyVector<Type_t, D> ret;
    for (unsigned d = 0; d < D; ++d)
      ret[d] = op(lhs[d], rhs[d]);
    return ret;
  }
};

template<class T1, class T2, class OP, unsigned D>
struct OTBinary<TinyVector<T1, D>, T2, OP>
{
  typedef typename BinaryReturn<T1, T2, OP>::Type_t Type_t;
  KOKKOS_INLINE_FUNCTION static TinyVector<Type_t, D>
  apply(const TinyVector<T1, D>& lhs, const T2& rhs, OP op)
  {
    TinyVector<Type_t, D> ret;
    for (unsigned d = 0; d < D; ++d)
      ret[d] = op(lhs[d], rhs);
    return ret;
  }
};

template<class T1, class T2, class OP, unsigned D>
struct OTBinary<T1, TinyVector<T2, D>, OP>
{
  typedef typename BinaryReturn<T1, T2, OP>::Type_t Type_t;
  KOKKOS_INLINE_FUNCTION static TinyVector<Type_t, D>
  apply(const T1& lhs, const TinyVector<T2, D>& rhs, OP op)
  {
    TinyVector<Type_t, D> ret;
    for (unsigned d = 0; d < D; ++d)
      ret[d] = op(lhs, rhs[d]);
    return ret;
  }
};


//////////////////////////////////////////////////////////////////////
//
// Specializations for TinyVector dot TinyVector
//
//////////////////////////////////////////////////////////////////////

template<class T1, class T2, unsigned D>
struct OTDot<TinyVector<T1, D>, TinyVector<T2, D>>
{
  typedef typename BinaryReturn<T1, T2, OpMultiply>::Type_t Type_t;
  KOKKOS_INLINE_FUNCTION static Type_t
  apply(const TinyVector<T1, D>& lhs, const TinyVector<T2, D>& rhs)
  {
    Type_t res = lhs[0] * rhs[0];
    for (unsigned d = 1; d < D; ++d)
      res += lhs[d] * rhs[d];
    return res;
  }
};

/** specialization for real-complex TinyVector */
template<class T1>
struct OTDot<TinyVector<T1, 3>, TinyVector<std::complex<T1>, 3>>
{
  typedef T1 Type_t;
  KOKKOS_INLINE_FUNCTION static Type_t
  apply(const TinyVector<T1, 3>& lhs, const TinyVector<std::complex<T1>, 3>& rhs)
  {
    return lhs[0] * rhs[0].real() + lhs[1] * rhs[1].real() + lhs[2] * rhs[2].real();
  }
};

/** specialization for complex-real TinyVector */
template<class T1, class T2>
struct OTDot<TinyVector<std::complex<T1>, 3>, TinyVector<T2, 3>>
{
  typedef T1 Type_t;
  KOKKOS_INLINE_FUNCTION static Type_t
  apply(const TinyVector<std::complex<T1>, 3>& lhs, const TinyVector<T2, 3>& rhs)
  {
    return lhs[0].real() * rhs[0] + lhs[1].real() * rhs[1] + lhs[2].real() * rhs[2];
  }
};

/** specialization for complex-complex TinyVector */
template<class T1, class T2>
struct OTDot<TinyVector<std::complex<T1>, 3>, TinyVector<std::complex<T2>, 3>>
{
  typedef typename BinaryReturn<std::complex<T1>, std::complex<T2>, OpMultiply>::Type_t Type_t;
  KOKKOS_INLINE_FUNCTION static Type_t
  apply(const TinyVector<std::complex<T1>, 3>& lhs, const TinyVector<std::complex<T2>, 3>& rhs)
  {
    return std::complex<T1>(lhs[0].real() * rhs[0].real() - lhs[0].imag() * rhs[0].imag() +
                                lhs[1].real() * rhs[1].real() - lhs[1].imag() * rhs[1].imag() +
                                lhs[2].real() * rhs[2].real() - lhs[2].imag() * rhs[2].imag(),
                            lhs[0].real() * rhs[0].imag() + lhs[0].imag() * rhs[0].real() +
                                lhs[1].real() * rhs[1].imag() + lhs[1].imag() * rhs[1].real() +
                                lhs[2].real() * rhs[2].imag() + lhs[2].imag() * rhs[2].real());
  }
};

//////////////////////////////////////////////////////////////////////
//
// Definition of the struct OTCross.
//
//////////////////////////////////////////////////////////////////////

template<class T1, class T2>
struct OTCross
{};

//////////////////////////////////////////////////////////////////////
//
// Specializations for TinyVector cross TinyVector
//
//////////////////////////////////////////////////////////////////////

template<class T1, class T2, unsigned D>
struct OTCross<TinyVector<T1, D>, TinyVector<T2, D>>
{
  typedef typename BinaryReturn<T1, T2, OpMultiply>::Type_t Type_t;
  KOKKOS_INLINE_FUNCTION static TinyVector<Type_t, D>
  apply(const TinyVector<T1, D>& a, const TinyVector<T2, D>& b)
  {
    TinyVector<Type_t, D> bogusCross(-99999);
    return bogusCross;
  }
};

template<class T1, class T2>
struct OTCross<TinyVector<T1, 3>, TinyVector<T2, 3>>
{
  typedef typename BinaryReturn<T1, T2, OpMultiply>::Type_t Type_t;
  KOKKOS_INLINE_FUNCTION static TinyVector<Type_t, 3>
  apply(const TinyVector<T1, 3>& a, const TinyVector<T2, 3>& b)
  {
    TinyVector<Type_t, 3> cross;
    cross[0] = a[1] * b[2] - a[2] * b[1];
    cross[1] = a[2] * b[0] - a[0] * b[2];
    cross[2] = a[0] * b[1] - a[1] * b[0];
    return cross;
  }
};

} // namespace qmcplusplus

#endif // OHMMS_TINYVECTOR_OPERATORS_H
