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

#ifndef OHMMS_OMPTINYVECTOR_H
#define OHMMS_OMPTINYVECTOR_H

// include files
#include <iomanip>
#include "OhmmsPETE/TinyVector.h"

namespace qmcplusplus
{
/** Fixed-size array. candidate for array<T,D>
 */
#pragma omp declare target
template <class T, unsigned D> struct OMPTinyVector
{
  typedef T Type_t;
  enum
  {
    Size = D
  };
  T X[Size];

  // Default Constructor initializes to zero.
  inline OMPTinyVector()
  {
    for (size_t d = 0; d < D; ++d) X[d] = 0;
  }

  // Templated OMPTinyVector constructor.
  template <class T1>
  inline OMPTinyVector(const TinyVector<T1, D> &rhs)
  {
    for (size_t d = 0; d < D; ++d) X[d] = rhs[d];
  }

  // Constructor from a single T
  inline OMPTinyVector(const T &x00)
  {
    for (size_t d = 0; d < D; ++d) X[d] = x00;
  }

  // Constructors for fixed dimension
  inline OMPTinyVector(const T &x00, const T &x01)
  {
    X[0] = x00;
    X[1] = x01;
  }
  inline OMPTinyVector(const T &x00, const T &x01, const T &x02)
  {
    X[0] = x00;
    X[1] = x01;
    X[2] = x02;
  }
  inline OMPTinyVector(const T &x00, const T &x01, const T &x02, const T &x03)
  {
    X[0] = x00;
    X[1] = x01;
    X[2] = x02;
    X[3] = x03;
  }

  inline OMPTinyVector(const T &x00, const T &x01, const T &x02, const T &x03,
                    const T &x10, const T &x11, const T &x12, const T &x13,
                    const T &x20, const T &x21, const T &x22, const T &x23,
                    const T &x30, const T &x31, const T &x32, const T &x33)
  {
    X[0]  = x00;
    X[1]  = x01;
    X[2]  = x02;
    X[3]  = x03;
    X[4]  = x10;
    X[5]  = x11;
    X[6]  = x12;
    X[7]  = x13;
    X[8]  = x20;
    X[9]  = x21;
    X[10] = x22;
    X[11] = x23;
    X[12] = x30;
    X[13] = x31;
    X[14] = x32;
    X[15] = x33;
  }

  inline OMPTinyVector(const T *restrict base, int offset)
  {
    #pragma unroll(D)
    for (int i = 0; i < D; ++i) X[i] = base[i * offset];
  }

  // Destructor
  ~OMPTinyVector() {}

  inline int size() const { return D; }

  inline int byteSize() const { return D * sizeof(T); }

  // Get and Set Operations
  inline Type_t &operator[](unsigned int i) { return X[i]; }
  inline const Type_t &operator[](unsigned int i) const { return X[i]; }

  inline Type_t *data() { return X; }
  inline const Type_t *data() const { return X; }
  inline Type_t *begin() { return X; }
  inline const Type_t *begin() const { return X; }
  inline Type_t *end() { return X + D; }
  inline const Type_t *end() const { return X + D; }
};
#pragma omp end declare target

//----------------------------------------------------------------------
// I/O
template <class T> struct printOMPTinyVector
{
};

// specialized for Vector<OMPTinyVector<T,D> >
template <class T, unsigned D> struct printOMPTinyVector<OMPTinyVector<T, D>>
{
  inline static void print(std::ostream &os, const OMPTinyVector<T, D> &r)
  {
    for (int d = 0; d < D; d++)
      os << std::setw(18) << std::setprecision(10) << r[d];
  }
};

// specialized for Vector<OMPTinyVector<T,2> >
template <class T> struct printOMPTinyVector<OMPTinyVector<T, 2>>
{
  inline static void print(std::ostream &os, const OMPTinyVector<T, 2> &r)
  {
    os << std::setw(18) << std::setprecision(10) << r[0] << std::setw(18)
       << std::setprecision(10) << r[1];
  }
};

// specialized for Vector<OMPTinyVector<T,3> >
template <class T> struct printOMPTinyVector<OMPTinyVector<T, 3>>
{
  inline static void print(std::ostream &os, const OMPTinyVector<T, 3> &r)
  {
    os << std::setw(18) << std::setprecision(10) << r[0] << std::setw(18)
       << std::setprecision(10) << r[1] << std::setw(18)
       << std::setprecision(10) << r[2];
  }
};

template <class T, unsigned D>
std::ostream &operator<<(std::ostream &out, const OMPTinyVector<T, D> &rhs)
{
  printOMPTinyVector<OMPTinyVector<T, D>>::print(out, rhs);
  return out;
}

template <class T, unsigned D>
std::istream &operator>>(std::istream &is, OMPTinyVector<T, D> &rhs)
{
  // printOMPTinyVector<OMPTinyVector<T,D> >::print(out,rhs);
  for (int i = 0; i < D; i++) is >> rhs[i];
  return is;
}
}

#endif // VEKTOR_H
