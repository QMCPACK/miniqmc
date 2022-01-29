//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2020 QMCPACK developers.
//
// File developed by: Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//                    Jeremy McMinnis, jmcminis@gmail.com, University of Illinois at Urbana-Champaign
//                    Mark A. Berrill, berrillma@ornl.gov, Oak Ridge National Laboratory
//
// File created by: Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//////////////////////////////////////////////////////////////////////////////////////


#ifndef QMCPLUSPLUS_CLOCK_H
#define QMCPLUSPLUS_CLOCK_H

#include <sys/time.h>
#include <stddef.h>
#include "Utilities/Configuration.h"

namespace qmcplusplus
{
/** functor for high precision clock
 * calling CPUClock()() returns the clock value
 */
class CPUClock
{
public:
  double operator()()
  {
#ifdef _OPENMP
    return omp_get_wtime();
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (1.e-6) * tv.tv_usec;
#endif
  }
};

} // namespace qmcplusplus
#endif
