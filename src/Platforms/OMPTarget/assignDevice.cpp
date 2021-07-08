//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2021 QMCPACK developers.
//
// File developed by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//
// File created by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//////////////////////////////////////////////////////////////////////////////////////


#include <stdexcept>
#include <iostream>
#include <omp.h>
#include "config.h"

namespace qmcplusplus
{
/** distribute MPI ranks among devices
 *
 * the amount of MPI ranks for each device differs by 1 at maximum.
 * larger id has more MPI ranks.
 */
int getDeviceID(int rank_id, int num_ranks, int num_devices)
{
  if (num_ranks < num_devices)
    num_devices = num_ranks;
  // ranks are equally distributed among devices
  int min_ranks_per_device = num_ranks / num_devices;
  int residual             = num_ranks % num_devices;
  int assigned_device_id;
  if (rank_id < min_ranks_per_device * (num_devices - residual))
    assigned_device_id = rank_id / min_ranks_per_device;
  else
    assigned_device_id = (rank_id + num_devices - residual) / (min_ranks_per_device + 1);
  return assigned_device_id;
}

void assignDevice(int& num_accelerators, int& assigned_accelerators_id, const int rank_id, const int num_ranks)
{
#if defined(ENABLE_OFFLOAD)
  int ompDeviceCount = omp_get_num_devices();
  int ompDeviceID;
  if (num_accelerators == 0)
    num_accelerators = ompDeviceCount;
  else if (num_accelerators != ompDeviceCount)
    throw std::runtime_error("Inconsistent number of OpenMP devices with the previous record!");
  if (ompDeviceCount > num_ranks && rank_id == 0)
    std::cerr << "More OpenMP devices than the number of MPI ranks. "
              << "Some devices will be left idle.\n"
              << "There is potential performance issue with the GPU affinity.\n";
  if (num_accelerators > 0)
  {
    ompDeviceID = getDeviceID(rank_id, num_ranks, ompDeviceCount);
    if (assigned_accelerators_id < 0)
      assigned_accelerators_id = ompDeviceID;
    else if (assigned_accelerators_id != ompDeviceID)
      throw std::runtime_error("Inconsistent assigned OpenMP devices with the previous record!");
    omp_set_default_device(ompDeviceID);
  }
#endif
}
}
