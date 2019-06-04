//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2019 QMCPACK developers.
//
// File developed by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//
// File created by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//////////////////////////////////////////////////////////////////////////////////////


#ifndef QMCPLUSPLUS_DETERMINANT_HELPER_H
#define QMCPLUSPLUS_DETERMINANT_HELPER_H

#include <cmath>

namespace qmcplusplus
{

template<typename T>
inline T evaluatePhase(T sign_v)
{
  return T((sign_v > 0) ? 0.0 : M_PI);
}

template<typename T>
inline T evaluatePhase(const std::complex<T>& psi)
{
  return T(std::arg(psi));
}

/** evaluate the log(|psi|) and phase
 * @param psi real/complex value
 * @param phase phase of psi
 * @return log(|psi|)
 */
template<class T>
inline T evaluateLogAndPhase(const T psi, T& phase)
{
  if (psi < 0.0)
  {
    phase = M_PI;
    return std::log(-psi);
  }
  else
  {
    phase = 0.0;
    return std::log(psi);
  }
}

template<class T>
inline T evaluateLogAndPhase(const std::complex<T>& psi, T& phase)
{
  phase = std::arg(psi);
  if (phase < 0.0)
    phase += 2.0 * M_PI;
  return std::log(std::abs(psi));
}

/** generic conversion from type T1 to type T2 using implicit conversion
*/
template<typename T1, typename T2>
inline void convert(const T1& in, T2& out)
{
  out = static_cast<T2>(in);
}

/** specialization of conversion from complex to real
*/
template<typename T1, typename T2>
inline void convert(const std::complex<T1>& in, T2& out)
{
  out = static_cast<T2>(in.real());
}

} // namespace qmcplusplus

#endif // QMCPLUSPLUS_DETERMINANT_HELPER_H
