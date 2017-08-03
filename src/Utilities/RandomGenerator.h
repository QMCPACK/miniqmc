//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by: Ken Esler, kpesler@gmail.com, University of Illinois at
//                    Urbana-Champaign
//                    Jeongnim Kim, jeongnim.kim@gmail.com, University of
//                    Illinois at Urbana-Champaign
//                    Jeremy McMinnis, jmcminis@gmail.com, University of
//                    Illinois at Urbana-Champaign
//                    Mark Dewing, markdewing@gmail.com, University of Illinois
//                    at Urbana-Champaign
//
// File created by: Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois
//                  at Urbana-Champaign
//////////////////////////////////////////////////////////////////////////////////////

/** @file RandomGenerator.h
 * @brief Declare a global Random Number Generator
 *
 * Selected among
 * - C++11 std::random
 * - (other choices are in the QMCPACK distribution)
 *
 * qmcplusplus::Random() returns a random number [0,1)
 *
 * When OpenMP is enabled, it is important to make sure each thread has its
 * own random number generator with a unique seed.  Using a global lock on
 *  a single generator would slow down the applications significantly.
 *
 */
#ifndef OHMMS_RANDOMGENERATOR
#define OHMMS_RANDOMGENERATOR
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <cmath>
#include <ctime>

#include <stdint.h>

struct BoxMuller2
{
  template <typename RNG>
  static inline void generate(RNG &rng, double *restrict a, int n)
  {
    for (int i = 0; i + 1 < n; i += 2)
    {
      double temp1 = 1.0 - 0.9999999999 * rng(), temp2 = rng();
      a[i]     = sqrt(-2.0 * log(temp1)) * cos(6.283185306 * temp2);
      a[i + 1] = sqrt(-2.0 * log(temp1)) * sin(6.283185306 * temp2);
    }
    if (n % 2 == 1)
    {
      double temp1 = 1 - 0.9999999999 * rng(), temp2 = rng();
      a[n - 1] = sqrt(-2.0 * log(temp1)) * cos(6.283185306 * temp2);
    }
  }

  template <typename RNG>
  static inline void generate(RNG &rng, float *restrict a, int n)
  {
    for (int i = 0; i + 1 < n; i += 2)
    {
      float temp1 = 1.0f - 0.9999999999f * rng(), temp2 = rng();
      a[i]     = sqrtf(-2.0f * logf(temp1)) * cosf(6.283185306f * temp2);
      a[i + 1] = sqrtf(-2.0f * logf(temp1)) * sinf(6.283185306f * temp2);
    }
    if (n % 2 == 1)
    {
      float temp1 = 1.0f - 0.9999999999f * rng(), temp2 = rng();
      a[n - 1] = sqrtf(-2.0f * logf(temp1)) * cosf(6.283185306f * temp2);
    }
  }
};

inline uint32_t MakeSeed(int i, int n)
{
  const uint32_t u = 1 << 10;
  return static_cast<uint32_t>(std::time(nullptr)) % u + (i + 1) * n + i;
}

#include "Utilities/StdRandom.h"
namespace qmcplusplus
{
template <class T> using RandomGenerator = StdRandom<T>;
typedef StdRandom<OHMMS_PRECISION_FULL> RandomGenerator_t;
extern RandomGenerator_t Random;
}

#endif
