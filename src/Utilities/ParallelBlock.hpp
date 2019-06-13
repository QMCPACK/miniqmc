////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2019 QMCPACK developers.
//
// File developed by:
// Peter Doak, doakpw@ornl.gov, Oak Ridge National Lab
//
// File created by:
// Peter Doak, doakpw@ornl.gov, Oak Ridge National Lab
////////////////////////////////////////////////////////////////////////////////

#ifndef QMCPLUSPLUS_PARALLELBLOCK_HPP
#define QMCPLUSPLUS_PARALLELBLOCK_HPP

#include <functional>
#include <thread>
#include <omp.h>
#include <iostream>

namespace qmcplusplus
{

enum class ParallelBlockThreading
{
  OPENMP,
  STD
  // Planned
  // Boost
  // hpx
};

/** Blocks until entire parallel block is done
 *
 *  Required to prevent end of QMC block synchronization occuring
 *  before all crowds are complete. For Devices with global or partially global
 *  synchronization on memory management operations. RAII deallocation 
 *  badly damage performance.
 */
template<ParallelBlockThreading TT>
class ParallelBlockBarrier
{
public:
  ParallelBlockBarrier(unsigned int num_threads) : num_threads_(num_threads) {}
  void wait();

private:
  unsigned int num_threads_;
};

template<ParallelBlockThreading TT>
inline void ParallelBlockBarrier<TT>::wait()
{
#pragma omp barrier
}

template<>
inline void ParallelBlockBarrier<ParallelBlockThreading::STD>::wait()
{
  std::cout << "Barrier not supported by std threading. Implementation needed to avoid performance hit at end of QMCBlock.\n";
}


/** Simple abstraction to launch num_threads running the same task
 *
 *  All run the same task and get the same arg set plus an "id" from 0 to num_threads - 1
 */
template<ParallelBlockThreading TT>
class ParallelBlock
{
public:
  ParallelBlock(unsigned int num_threads) : num_threads_(num_threads) {}
  template<typename F, typename... Args>
  void operator()(F&& f, Args&&... args);
  template<typename F, typename... Args>
  void operator()(F&& f, ParallelBlockBarrier<TT>& barrier, Args&&... args);

private:
  unsigned int num_threads_;
};

template<ParallelBlockThreading TT>
template<typename F, typename... Args>
void ParallelBlock<TT>::operator()(F&& f, Args&&... args)

{
#pragma omp parallel for
  for (int task_id = 0; task_id < num_threads_; ++task_id)
  {
    f(task_id, std::forward<Args>(args)...);
  }
}
    
template<>
template<typename F, typename... Args>
void ParallelBlock<ParallelBlockThreading::STD>::operator()(F&& f, Args&&... args)

{
  std::vector<std::thread> threads(num_threads_);

  for (int task_id = 0; task_id < num_threads_; ++task_id)
  {
      threads[task_id] = std::thread(std::forward<F>(f), task_id, std::forward<Args>(args)...);
  }

  for (int task_id = 0; task_id < num_threads_; ++task_id)
  {
    threads[task_id].join();
  }
}

template<ParallelBlockThreading TT>
template<typename F, typename... Args>
void ParallelBlock<TT>::operator()(F&& f, ParallelBlockBarrier<TT>& barrier, Args&&... args)

{
  omp_set_num_threads(num_threads_);
#pragma omp parallel for
  for (int task_id = 0; task_id < num_threads_; ++task_id)
  {
    f(task_id, barrier, std::forward<Args>(args)...);
  }
}

} // namespace qmcplusplus

#endif
