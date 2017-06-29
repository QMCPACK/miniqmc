//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by: Ken Esler, kpesler@gmail.com, University of Illinois at Urbana-Champaign
//                    Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//                    Jeremy McMinnis, jmcminis@gmail.com, University of Illinois at Urbana-Champaign
//                    Mark A. Berrill, berrillma@ornl.gov, Oak Ridge National Laboratory
//
// File created by: Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//////////////////////////////////////////////////////////////////////////////////////
    
    
/** @file BasisSetBase.h
 * @brief Declaration of a base class of BasisSet
 */
#ifndef QMCPLUSPLUS_ORBITALSETTRAITS_H
#define QMCPLUSPLUS_ORBITALSETTRAITS_H

#include "Configuration.h"
#include "Numerics/VectorViewer.h"

namespace qmcplusplus
{

template<class T> struct scalar_traits
{
  enum {DIM=1};
  typedef T real_type;
  typedef T value_type;
};

template<typename T>
struct scalar_traits<std::complex<T> >
{
  enum {DIM=2};
  typedef T          real_type;
  typedef std::complex<T> value_type;
};

/** trait class to handel a set of Orbitals
 */
template<typename T>
struct OrbitalSetTraits//: public OrbitalTraits<T>
{
  enum {DIM=OHMMS_DIM};
  typedef typename scalar_traits <T>::real_type RealType;
  typedef typename scalar_traits <T>::value_type ValueType;
  typedef int                            IndexType;
  typedef TinyVector<RealType,DIM>       PosType;
  typedef TinyVector<ValueType,DIM>      GradType;
  typedef Tensor<ValueType,DIM>          HessType;
  typedef Tensor<ValueType,DIM>          TensorType;
  typedef TinyVector<Tensor<ValueType,DIM>,DIM> GradHessType;
  typedef Vector<IndexType>     IndexVector_t;
  typedef Vector<ValueType>     ValueVector_t;
  typedef Matrix<ValueType>     ValueMatrix_t;
  typedef Vector<GradType>      GradVector_t;
  typedef Matrix<GradType>      GradMatrix_t;
  typedef Vector<HessType>      HessVector_t;
  typedef Matrix<HessType>      HessMatrix_t;
  typedef Vector<GradHessType>  GradHessVector_t;
  typedef Matrix<GradHessType>  GradHessMatrix_t;
  typedef VectorViewer<ValueType>             RefVector_t;
  typedef VectorSoaContainer<ValueType,DIM+2> VGLVector_t;
};

template<typename T> inline T evaluatePhase(T sign_v)
{
  return (T)((sign_v>0)?0.0:M_PI);
}

template<typename T> inline T evaluatePhase(const std::complex<T>& psi)
{
  return (T)(std::arg(psi));
}

/** evaluate the log(|psi|) and phase
 * @param psi real/complex value
 * @param phase phase of psi
 * @return log(|psi|)
 */
template<class T>
inline T evaluateLogAndPhase(const T psi, T& phase)
{
  if(psi<0.0)
  {
    phase= M_PI;
    return std::log(-psi);
  }
  else
  {
    phase = 0.0;
    return std::log(psi);
  }
}

template<class T>
inline T
evaluateLogAndPhase(const std::complex<T>& psi, T& phase)
{
  phase = std::arg(psi);
  if(phase<0.0)
    phase += 2.0*M_PI;
  return std::log( std::abs(psi) );
//      return 0.5*std::log(psi.real()*psi.real()+psi.imag()*psi.imag());
  //return std::log(psi);
}

inline double evaluatePhase(const double psi)
{
  return (psi<std::numeric_limits<double>::epsilon())?M_PI:0.0;
}

inline double evaluatePhase(const std::complex<double>& psi)
{
  return std::arg(psi);
}

}

#endif
