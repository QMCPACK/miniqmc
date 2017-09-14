////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by:
// Jeongnim Kim, jeongnim.kim@intel.com,
//    Intel Corp.
// Amrita Mathuriya, amrita.mathuriya@intel.com,
//    Intel Corp.
//
// File created by:
// Jeongnim Kim, jeongnim.kim@intel.com,
//    Intel Corp.
////////////////////////////////////////////////////////////////////////////////
// -*- C++ -*-
/** @file check_spo.cpp
 * @brief Miniapp to check 3D spline implementation against the reference.
 */
#include <Configuration.h>
#include "OMP_target_test/OMPVector.h"
#include "OMP_target_test/OMPTinyVector.h"

using namespace std;
using namespace qmcplusplus;

int main(int argc, char **argv)
{

  OhmmsInfo("check_spo");
  const int len = 12;

  std::vector<OMPVector<int> > vec_th(omp_get_max_threads());
  OMPVector<int *> shadow(omp_get_max_threads());
  #pragma omp parallel
  vec_th[omp_get_thread_num()].resize(len);

  int **restrict shadows_ptr=shadow.data();
  for(size_t tid=0; tid<shadow.size(); tid++)
  {
    int *restrict vec_ptr=vec_th[tid].data();
    #pragma omp target map(to:tid)
    {
      shadows_ptr[tid]=vec_ptr;
    }
  }

  const size_t nt=shadow.size();
  #pragma omp target teams distribute map(to:nt)
  for(size_t iw=0; iw<nt; iw++)
  {
    #pragma omp parallel for
    for(size_t iel=0; iel<len; iel++)
      shadows_ptr[iw][iel] = iel+iw;
  }

  for(size_t tid=0; tid<shadow.size(); tid++)
  {
    vec_th[tid].update_from_device();
    std::cout << "iw = " << tid << " : ";
    for(size_t iel=0; iel<len; iel++)
      std::cout << "  " << vec_th[tid][iel];
    std::cout << std::endl;
  }

  return 0;
}
