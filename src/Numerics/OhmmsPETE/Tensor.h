////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by:
// Ken Esler, kpesler@gmail.com,
//    University of Illinois at Urbana-Champaign
// Jeremy McMinnis, jmcminis@gmail.com,
//    University of Illinois at Urbana-Champaign
// Miguel Morales, moralessilva2@llnl.gov,
//    Lawrence Livermore National Laboratory
// Jeongnim Kim, jeongnim.kim@gmail.com,
//    University of Illinois at Urbana-Champaign
//
// File created by:
// Jeongnim Kim, jeongnim.kim@gmail.com,
//    University of Illinois at Urbana-Champaign
////////////////////////////////////////////////////////////////////////////////

#ifndef OHMMS_TENSOR_H
#define OHMMS_TENSOR_H

#include "Numerics/PETE/PETE.h"
#include "Numerics/OhmmsPETE/OhmmsTinyMeta.h"

/***************************************************************************
 *
 * The POOMA Framework
 *
 * This program was prepared by the Regents of the University of
 * California at Los Alamos National Laboratory (the University) under
 * Contract No.  W-7405-ENG-36 with the U.S. Department of Energy (DOE).
 * The University has certain rights in the program pursuant to the
 * contract and the program should not be copied or distributed outside
 * your organization.  All rights in the program are reserved by the DOE
 * and the University.  Neither the U.S.  Government nor the University
 * makes any warranty, express or implied, or assumes any liability or
 * responsibility for the use of this software
 *
 * Visit http://www.acl.lanl.gov/POOMA for more details
 *
 ***************************************************************************/

namespace qmcplusplus
{

/** Tensor<T,D>  class for D by D tensor
 *
 * @tparam T datatype
 * @tparm D dimension
 */
template <class T, unsigned D> class Tensor
{
public:
  typedef T Type_t;
  enum
  {
    ElemDim = 2
  };
  enum
  {
    Size = D * D
  };

  // Default Constructor
  KOKKOS_INLINE_FUNCTION
  Tensor()
  {
    OTAssign<Tensor<T, D>, T, OpAssign>::apply(*this, T(0), OpAssign());
  }

  // A noninitializing ctor.
  class DontInitialize
  {
  };
  KOKKOS_INLINE_FUNCTION
  Tensor(DontInitialize) {}

  // Copy Constructor
  KOKKOS_INLINE_FUNCTION
  Tensor(const Tensor<T, D> &rhs)
  {
    OTAssign<Tensor<T, D>, Tensor<T, D>, OpAssign>::apply(*this, rhs,
                                                          OpAssign());
  }

  // constructor from a single T
  KOKKOS_INLINE_FUNCTION
  Tensor(const T &x00)
  {
    OTAssign<Tensor<T, D>, T, OpAssign>::apply(*this, x00, OpAssign());
  }

  // constructors for fixed dimension
  KOKKOS_INLINE_FUNCTION
  Tensor(const T &x00, const T &x10, const T &x01, const T &x11)
  {
    X[0] = x00;
    X[1] = x10;
    X[2] = x01;
    X[3] = x11;
  }
  KOKKOS_INLINE_FUNCTION
  Tensor(const T &x00, const T &x10, const T &x20, const T &x01, const T &x11,
         const T &x21, const T &x02, const T &x12, const T &x22)
  {
    X[0] = x00;
    X[1] = x10;
    X[2] = x20;
    X[3] = x01;
    X[4] = x11;
    X[5] = x21;
    X[6] = x02;
    X[7] = x12;
    X[8] = x22;
  }

  // destructor
  KOKKOS_INLINE_FUNCTION
  ~Tensor(){};

  // assignment operators
  KOKKOS_INLINE_FUNCTION Tensor<T, D> &operator=(const Tensor<T, D> &rhs)
  {
    OTAssign<Tensor<T, D>, Tensor<T, D>, OpAssign>::apply(*this, rhs,
                                                          OpAssign());
    return *this;
  }

  template <class T1> KOKKOS_INLINE_FUNCTION Tensor<T, D> &operator=(const Tensor<T1, D> &rhs)
  {
    OTAssign<Tensor<T, D>, Tensor<T1, D>, OpAssign>::apply(*this, rhs,
                                                           OpAssign());
    return *this;
  }
  KOKKOS_INLINE_FUNCTION Tensor<T, D> &operator=(const T &rhs)
  {
    OTAssign<Tensor<T, D>, T, OpAssign>::apply(*this, rhs, OpAssign());
    return *this;
  }

  // accumulation operators
  template <class T1> KOKKOS_INLINE_FUNCTION Tensor<T, D> &operator+=(const Tensor<T1, D> &rhs)
  {
    OTAssign<Tensor<T, D>, Tensor<T1, D>, OpAddAssign>::apply(*this, rhs,
                                                              OpAddAssign());
    return *this;
  }
  KOKKOS_INLINE_FUNCTION Tensor<T, D> &operator+=(const T &rhs)
  {
    OTAssign<Tensor<T, D>, T, OpAddAssign>::apply(*this, rhs, OpAddAssign());
    return *this;
  }

  template <class T1> KOKKOS_INLINE_FUNCTION Tensor<T, D> &operator-=(const Tensor<T1, D> &rhs)
  {
    OTAssign<Tensor<T, D>, Tensor<T1, D>, OpSubtractAssign>::apply(
        *this, rhs, OpSubtractAssign());
    return *this;
  }

  KOKKOS_INLINE_FUNCTION Tensor<T, D> &operator-=(const T &rhs)
  {
    OTAssign<Tensor<T, D>, T, OpSubtractAssign>::apply(*this, rhs,
                                                       OpSubtractAssign());
    return *this;
  }

  template <class T1> KOKKOS_INLINE_FUNCTION Tensor<T, D> &operator*=(const Tensor<T1, D> &rhs)
  {
    OTAssign<Tensor<T, D>, Tensor<T1, D>, OpMultiplyAssign>::apply(
        *this, rhs, OpMultiplyAssign());
    return *this;
  }

  KOKKOS_INLINE_FUNCTION Tensor<T, D> &operator*=(const T &rhs)
  {
    OTAssign<Tensor<T, D>, T, OpMultiplyAssign>::apply(*this, rhs,
                                                       OpMultiplyAssign());
    return *this;
  }

  template <class T1> KOKKOS_INLINE_FUNCTION Tensor<T, D> &operator/=(const Tensor<T1, D> &rhs)
  {
    OTAssign<Tensor<T, D>, Tensor<T1, D>, OpDivideAssign>::apply(
        *this, rhs, OpDivideAssign());
    return *this;
  }

  KOKKOS_INLINE_FUNCTION Tensor<T, D> &operator/=(const T &rhs)
  {
    OTAssign<Tensor<T, D>, T, OpDivideAssign>::apply(*this, rhs,
                                                     OpDivideAssign());
    return *this;
  }

  // Methods

  KOKKOS_INLINE_FUNCTION void diagonal(const T &rhs)
  {
    for (int i = 0; i < D; i++) (*this)(i, i) = rhs;
  }

  KOKKOS_INLINE_FUNCTION void add2diagonal(T rhs)
  {
    for (int i = 0; i < D; i++) (*this)(i, i) += rhs;
  }

  /// return the size
  KOKKOS_INLINE_FUNCTION int len() const { return Size; }
  /// return the size
  KOKKOS_INLINE_FUNCTION int size() const { return Size; }

  /** return the i-th value or assign
   * @param i index [0,D*D)
   */
  KOKKOS_INLINE_FUNCTION Type_t &operator[](unsigned int i) { return X[i]; }

  /** return the i-th value
   * @param i index [0,D*D)
   */
  KOKKOS_INLINE_FUNCTION Type_t operator[](unsigned int i) const { return X[i]; }

  // TJW: add these 12/16/97 to help with NegReflectAndZeroFace BC:
  // These are the same as operator[] but with () instead:
  KOKKOS_INLINE_FUNCTION Type_t &operator()(unsigned int i) { return X[i]; }

  KOKKOS_INLINE_FUNCTION Type_t operator()(unsigned int i) const { return X[i]; }
  // TJW.

  /** return the (i,j) component
   * @param i index [0,D)
   * @param j index [0,D)
   */
  KOKKOS_INLINE_FUNCTION Type_t operator()(unsigned int i, unsigned int j) const
  {
    return X[i * D + j];
  }

  /** return/assign the (i,j) component
   * @param i index [0,D)
   * @param j index [0,D)
   */
  KOKKOS_INLINE_FUNCTION Type_t &operator()(unsigned int i, unsigned int j)
  {
    return X[i * D + j];
  }

  KOKKOS_INLINE_FUNCTION TinyVector<T, D> getRow(unsigned int i)
  {
    TinyVector<T, D> res;
    for (int j = 0; j < D; j++) res[j] = X[i * D + j];
    return res;
  }

  KOKKOS_INLINE_FUNCTION TinyVector<T, D> getColumn(unsigned int i)
  {
    TinyVector<T, D> res;
    for (int j = 0; j < D; j++) res[j] = X[j * D + i];
    return res;
  }

  KOKKOS_INLINE_FUNCTION Type_t *data() { return X; }
  KOKKOS_INLINE_FUNCTION const Type_t *data() const { return X; }
  KOKKOS_INLINE_FUNCTION Type_t *begin() { return X; }
  KOKKOS_INLINE_FUNCTION const Type_t *begin() const { return X; }
  KOKKOS_INLINE_FUNCTION Type_t *end() { return X + Size; }
  KOKKOS_INLINE_FUNCTION const Type_t *end() const { return X + Size; }

private:
  // The elements themselves.
  T X[Size];
};

//////////////////////////////////////////////////////////////////////
//
// Free functions
//
//////////////////////////////////////////////////////////////////////

/** trace \f$ result = \sum_k rhs(k,k)\f$
 * @param rhs a tensor
 */
template <class T, unsigned D> KOKKOS_INLINE_FUNCTION T trace(const Tensor<T, D> &rhs)
{
  T result = 0.0;
  for (int i = 0; i < D; i++) result += rhs(i, i);
  return result;
}

/** transpose a tensor
 */
template <class T, unsigned D>
KOKKOS_INLINE_FUNCTION Tensor<T, D> transpose(const Tensor<T, D> &rhs)
{
  Tensor<T, D> result; // = Tensor<T,D>::DontInitialize();
  for (int j = 0; j < D; j++)
    for (int i = 0; i < D; i++) result(i, j) = rhs(j, i);
  return result;
}

/** Tr(a*b), \f$ \sum_i\sum_j a(i,j)*b(j,i) \f$
 */
template <class T1, class T2, unsigned D>
KOKKOS_INLINE_FUNCTION T1 trace(const Tensor<T1, D> &a, const Tensor<T2, D> &b)
{
  T1 result = 0.0;
  for (int i = 0; i < D; i++)
    for (int j = 0; j < D; j++) result += a(i, j) * b(j, i);
  return result;
}

/** Tr(a^t *b), \f$ \sum_i\sum_j a(i,j)*b(i,j) \f$
 */
template <class T, unsigned D>
KOKKOS_INLINE_FUNCTION T traceAtB(const Tensor<T, D> &a, const Tensor<T, D> &b)
{
  T result = 0.0;
  for (int i = 0; i < D * D; i++) result += a(i) * b(i);
  return result;
}

/** Tr(a^t *b), \f$ \sum_i\sum_j a(i,j)*b(i,j) \f$
 */
template <class T1, class T2, unsigned D>
KOKKOS_INLINE_FUNCTION typename BinaryReturn<T1, T2, OpMultiply>::Type_t
traceAtB(const Tensor<T1, D> &a, const Tensor<T2, D> &b)
{
  typedef typename BinaryReturn<T1, T2, OpMultiply>::Type_t T;
  T result = 0.0;
  for (int i = 0; i < D * D; i++) result += a(i) * b(i);
  return result;
}

/// Binary Operators
OHMMS_META_BINARY_OPERATORS(Tensor, operator+, OpAdd)
OHMMS_META_BINARY_OPERATORS(Tensor, operator-, OpSubtract)
OHMMS_META_BINARY_OPERATORS(Tensor, operator*, OpMultiply)
OHMMS_META_BINARY_OPERATORS(Tensor, operator/, OpDivide)

/** Tensor-Tensor dot product \f$result(i,j)=\sum_k lhs(i,k)*rhs(k,j)\f$
 * @param lhs  a tensor
 * @param rhs  a tensor
 */
template <class T1, class T2, unsigned D>
KOKKOS_INLINE_FUNCTION Tensor<typename BinaryReturn<T1, T2, OpMultiply>::Type_t, D>
dot(const Tensor<T1, D> &lhs, const Tensor<T2, D> &rhs)
{
  return OTDot<Tensor<T1, D>, Tensor<T2, D>>::apply(lhs, rhs);
}

/** Vector-Tensor dot product \f$result(i)=\sum_k lhs(k)*rhs(k,i)\f$
 * @param lhs  a vector
 * @param rhs  a tensor
 */
template <class T1, class T2, unsigned D>
KOKKOS_INLINE_FUNCTION TinyVector<typename BinaryReturn<T1, T2, OpMultiply>::Type_t, D>
dot(const TinyVector<T1, D> &lhs, const Tensor<T2, D> &rhs)
{
  return OTDot<TinyVector<T1, D>, Tensor<T2, D>>::apply(lhs, rhs);
}

/** Tensor-Vector dot product \f$result(i)=\sum_k lhs(i,k)*rhs(k)\f$
 * @param lhs  a tensor
 * @param rhs  a vector
 */
template <class T1, class T2, unsigned D>
KOKKOS_INLINE_FUNCTION TinyVector<typename BinaryReturn<T1, T2, OpMultiply>::Type_t, D>
dot(const Tensor<T1, D> &lhs, const TinyVector<T2, D> &rhs)
{
  return OTDot<Tensor<T1, D>, TinyVector<T2, D>>::apply(lhs, rhs);
}

//----------------------------------------------------------------------
// I/O
template <class T, unsigned D>
std::ostream &operator<<(std::ostream &out, const Tensor<T, D> &rhs)
{
  if (D >= 1)
  {
    for (int i = 0; i < D; i++)
    {
      for (int j = 0; j < D - 1; j++)
      {
        out << rhs(i, j) << "  ";
      }
      out << rhs(i, D - 1) << " ";
      if (i < D - 1) out << std::endl;
    }
  }
  else
  {
    out << " " << rhs(0, 0) << " ";
  }
  return out;
}

template <class T, unsigned D>
std::istream &operator>>(std::istream &is, Tensor<T, D> &rhs)
{
  for (int i = 0; i < D * D; i++) is >> rhs[i];
  return is;
}
}

#endif // OHMMS_TENSOR_H
