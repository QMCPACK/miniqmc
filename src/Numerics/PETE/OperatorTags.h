// -*- C++ -*-
// ACL:license
// ----------------------------------------------------------------------
// This software and ancillary information (herein called "SOFTWARE")
// called PETE (Portable Expression Template Engine) is
// made available under the terms described here.  The SOFTWARE has been
// approved for release with associated LA-CC Number LA-CC-99-5.
//
// Unless otherwise indicated, this SOFTWARE has been authored by an
// employee or employees of the University of California, operator of the
// Los Alamos National Laboratory under Contract No.  W-7405-ENG-36 with
// the U.S. Department of Energy.  The U.S. Government has rights to use,
// reproduce, and distribute this SOFTWARE. The public may copy, distribute,
// prepare derivative works and publicly display this SOFTWARE without
// charge, provided that this Notice and any statement of authorship are
// reproduced on all copies.  Neither the Government nor the University
// makes any warranty, express or implied, or assumes any liability or
// responsibility for the use of this SOFTWARE.
//
// If SOFTWARE is modified to produce derivative works, such modified
// SOFTWARE should be clearly marked, so as not to confuse it with the
// version available from LANL.
//
// For more information about PETE, send e-mail to pete@acl.lanl.gov,
// or visit the PETE web page at http://www.acl.lanl.gov/pete/.
// ----------------------------------------------------------------------
// ACL:license

#include <cstdlib>
#include <cmath>
#include "clean_inlining.h"

#ifndef PETE_PETE_OPERATORTAGS_H
#define PETE_PETE_OPERATORTAGS_H

namespace qmcplusplus
{
struct OpAdd
{
  PETE_EMPTY_CONSTRUCTORS(OpAdd)
  template<class T1, class T2>
  inline typename BinaryReturn<T1, T2, OpAdd>::Type_t operator()(const T1& a, const T2& b) const
  {
    return (a + b);
  }
};

struct OpSubtract
{
  PETE_EMPTY_CONSTRUCTORS(OpSubtract)
  template<class T1, class T2>
  inline typename BinaryReturn<T1, T2, OpSubtract>::Type_t operator()(const T1& a, const T2& b) const
  {
    return (a - b);
  }
};

struct OpMultiply
{
  PETE_EMPTY_CONSTRUCTORS(OpMultiply)
  template<class T1, class T2>
  inline typename BinaryReturn<T1, T2, OpMultiply>::Type_t operator()(const T1& a, const T2& b) const
  {
    return (a * b);
  }
};

struct OpDivide
{
  PETE_EMPTY_CONSTRUCTORS(OpDivide)
  template<class T1, class T2>
  inline typename BinaryReturn<T1, T2, OpDivide>::Type_t operator()(const T1& a, const T2& b) const
  {
    return (a / b);
  }
};

struct OpMod
{
  PETE_EMPTY_CONSTRUCTORS(OpMod)
  template<class T1, class T2>
  inline typename BinaryReturn<T1, T2, OpMod>::Type_t operator()(const T1& a, const T2& b) const
  {
    return (a % b);
  }
};


struct OpAddAssign
{
  PETE_EMPTY_CONSTRUCTORS(OpAddAssign)
  template<class T1, class T2>
  inline typename BinaryReturn<T1, T2, OpAddAssign>::Type_t operator()(const T1& a, const T2& b) const
  {
    (const_cast<T1&>(a) += b);
    return const_cast<T1&>(a);
  }
};

template<class T1, class T2>
struct BinaryReturn<T1, T2, OpAddAssign>
{
  typedef T1& Type_t;
};

struct OpSubtractAssign
{
  PETE_EMPTY_CONSTRUCTORS(OpSubtractAssign)
  template<class T1, class T2>
  inline typename BinaryReturn<T1, T2, OpSubtractAssign>::Type_t
  operator()(const T1& a, const T2& b) const
  {
    (const_cast<T1&>(a) -= b);
    return const_cast<T1&>(a);
  }
};

template<class T1, class T2>
struct BinaryReturn<T1, T2, OpSubtractAssign>
{
  typedef T1& Type_t;
};

struct OpMultiplyAssign
{
  PETE_EMPTY_CONSTRUCTORS(OpMultiplyAssign)
  template<class T1, class T2>
  inline typename BinaryReturn<T1, T2, OpMultiplyAssign>::Type_t
  operator()(const T1& a, const T2& b) const
  {
    (const_cast<T1&>(a) *= b);
    return const_cast<T1&>(a);
  }
};

template<class T1, class T2>
struct BinaryReturn<T1, T2, OpMultiplyAssign>
{
  typedef T1& Type_t;
};

struct OpDivideAssign
{
  PETE_EMPTY_CONSTRUCTORS(OpDivideAssign)
  template<class T1, class T2>
  inline typename BinaryReturn<T1, T2, OpDivideAssign>::Type_t
  operator()(const T1& a, const T2& b) const
  {
    (const_cast<T1&>(a) /= b);
    return const_cast<T1&>(a);
  }
};

template<class T1, class T2>
struct BinaryReturn<T1, T2, OpDivideAssign>
{
  typedef T1& Type_t;
};

struct OpModAssign
{
  PETE_EMPTY_CONSTRUCTORS(OpModAssign)
  template<class T1, class T2>
  inline typename BinaryReturn<T1, T2, OpModAssign>::Type_t operator()(const T1& a, const T2& b) const
  {
    (const_cast<T1&>(a) %= b);
    return const_cast<T1&>(a);
  }
};

template<class T1, class T2>
struct BinaryReturn<T1, T2, OpModAssign>
{
  typedef T1& Type_t;
};


struct OpAssign
{
  PETE_EMPTY_CONSTRUCTORS(OpAssign)
  template<class T1, class T2>
  KOKKOS_INLINE_FUNCTION typename BinaryReturn<T1, T2, OpAssign>::Type_t operator()(const T1& a, const T2& b) const
  {
    return (const_cast<T1&>(a) = b);
  }
};

template<class T1, class T2>
struct BinaryReturn<T1, T2, OpAssign>
{
  typedef T1& Type_t;
};

} // namespace qmcplusplus
#endif // PETE_PETE_OPERATORTAGS_H
