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

#include <atomic>
#include <thread>
#include "catch.hpp"
#include "Utilities/ParallelBlock.hpp"

/** @file
 *
 *  We assume omp and std threading are always available.
 *  Boost::HANA would simplify this...
 *  i.e. one test could be written for all the implementations
 */

namespace qmcplusplus
{


template<ParallelBlockThreading TT>
struct testTaskBarrier
{
    static void test(const int ip,
	       ParallelBlockBarrier<TT>& barrier,
	       std::atomic<int>& counter)
{
  counter.fetch_add(1);
}
};
    
template<ParallelBlockThreading TT>
struct testTask
{
    static void test(const int ip,
	       std::atomic<int>& counter)
{
  counter.fetch_add(1);
}
};
		   
    
TEST_CASE("ParallelBlock OPENMP with Block Barrier", "[Utilities]") {
  int threads = 8;
  constexpr ParallelBlockThreading DT = ParallelBlockThreading::OPENMP;
  ParallelBlock<DT> par_block(threads);
  ParallelBlockBarrier<DT> barrier(threads);
  std::atomic<int> counter;
  counter = 0;
  par_block(testTaskBarrier<DT>::test, barrier, counter);
  REQUIRE(counter == 8);
}

TEST_CASE("ParallelBlock OPENMP", "[Utilities]") {
  int threads = 8;
  constexpr ParallelBlockThreading DT = ParallelBlockThreading::OPENMP;
  ParallelBlock<DT> par_block(threads);
  std::atomic<int> counter;
  counter = 0;
  par_block(testTask<DT>::test, counter);
  REQUIRE(counter == 8);
}

TEST_CASE("ParallelBlock std::thread", "[Utilities]") {
  int threads = 8;
  constexpr ParallelBlockThreading DTS = ParallelBlockThreading::STD;
  ParallelBlock<DTS> par_block(threads);
  std::atomic<int> counter;
  counter = 0;
  par_block(testTask<DTS>::test, std::ref(counter));
  REQUIRE(counter == 8);
}

}

