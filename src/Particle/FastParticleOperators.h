////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by:
// Jeongnim Kim, jeongnim.kim@gmail.com,
//    University of Illinois at Urbana-Champaign
// Jeremy McMinnis, jmcminis@gmail.com,
//    University of Illinois at Urbana-Champaign
// Mark A. Berrill, berrillma@ornl.gov,
//    Oak Ridge National Laboratory
//
// File created by:
// Jeongnim Kim, jeongnim.kim@gmail.com,
//    University of Illinois at Urbana-Champaign
////////////////////////////////////////////////////////////////////////////////

/** @file FastParticleOperators.h
 * @brief template functions to support conversion and handling of boundary
 * conditions.
 */
#ifndef OHMMS_FAST_PARTICLE_OPERATORS_H
#define OHMMS_FAST_PARTICLE_OPERATORS_H

namespace qmcplusplus
{
/** Dummy template class to be specialized
 *
 * - T1 the datatype to be transformed
 * - T2 the transformation matrix
 * - ORTHO true, if only Diagonal Elements are used
 */
template <class T1, class T2, unsigned D, bool ORTHO> struct ConvertPosUnit
{
};

template <class T>
struct ConvertPosUnit<ParticleAttrib<TinyVector<T, 3>>, Tensor<T, 3>, 3, false>
{

  typedef ParticleAttrib<TinyVector<T, 3>> Array_t;
  typedef Tensor<T, 3> Transformer_t;

  inline static void apply(const Array_t &pin, const Transformer_t &X,
                           Array_t &pout, int first, int last)
  {
    register T x00 = X[0], x01 = X[1], x02 = X[2], x10 = X[3], x11 = X[4],
               x12 = X[5], x20 = X[6], x21 = X[7], x22 = X[8];
#pragma ivdep
    for (int i = first; i < last; i++)
    {
      pout[i][0] = pin[i][0] * x00 + pin[i][1] * x10 + pin[i][2] * x20;
      pout[i][1] = pin[i][0] * x01 + pin[i][1] * x11 + pin[i][2] * x21;
      pout[i][2] = pin[i][0] * x02 + pin[i][1] * x12 + pin[i][2] * x22;
    }
  }

  inline static void apply(const Transformer_t &X, const Array_t &pin,
                           Array_t &pout, int first, int last)
  {
    register T x00 = X[0], x01 = X[1], x02 = X[2], x10 = X[3], x11 = X[4],
               x12 = X[5], x20 = X[6], x21 = X[7], x22 = X[8];
#pragma ivdep
    for (int i = first; i < last; i++)
    {
      pout[i][0] = pin[i][0] * x00 + pin[i][1] * x01 + pin[i][2] * x02;
      pout[i][1] = pin[i][0] * x10 + pin[i][1] * x11 + pin[i][2] * x12;
      pout[i][2] = pin[i][0] * x20 + pin[i][1] * x21 + pin[i][2] * x22;
    }
  }

  inline static void apply(Array_t &pinout, const Transformer_t &X, int first,
                           int last)
  {
    register T x00 = X[0], x01 = X[1], x02 = X[2], x10 = X[3], x11 = X[4],
               x12 = X[5], x20 = X[6], x21 = X[7], x22 = X[8];
#pragma ivdep
    for (int i = first; i < last; i++)
    {
      T _x(pinout[i][0]), _y(pinout[i][1]), _z(pinout[i][2]);
      pinout[i][0] = _x * x00 + _y * x10 + _z * x20;
      pinout[i][1] = _x * x01 + _y * x11 + _z * x21;
      pinout[i][2] = _x * x02 + _y * x12 + _z * x22;
    }
  }

  inline static void apply(const Transformer_t &X, Array_t &pinout, int first,
                           int last)
  {
    register T x00 = X[0], x01 = X[1], x02 = X[2], x10 = X[3], x11 = X[4],
               x12 = X[5], x20 = X[6], x21 = X[7], x22 = X[8];
#pragma ivdep
    for (int i = first; i < last; i++)
    {
      T _x(pinout[i][0]), _y(pinout[i][1]), _z(pinout[i][2]);
      pinout[i][0] = _x * x00 + _y * x01 + _z * x02;
      pinout[i][1] = _x * x10 + _y * x11 + _z * x12;
      pinout[i][2] = _x * x20 + _y * x21 + _z * x22;
    }
  }
};
}
#endif
