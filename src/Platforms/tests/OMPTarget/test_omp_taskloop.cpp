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


#include "catch.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include <memory>
#include "config.h"

namespace qmcplusplus
{

template<typename T>
class RefVectorWithLeader : public std::vector<std::reference_wrapper<T>>
{
public:
  RefVectorWithLeader(T& leader) : leader_(leader) {}

  RefVectorWithLeader(T& leader, const std::vector<std::reference_wrapper<T>>& vec) : leader_(leader)
  {
    for (T& element : vec)
      this->push_back(element);
  }

  T& getLeader() const { return leader_; }

  T& operator[](size_t i) const { return std::vector<std::reference_wrapper<T>>::operator[](i).get(); }

  template<typename CASTTYPE>
  CASTTYPE& getCastedLeader() const
  {
    static_assert(std::is_const<T>::value == std::is_const<CASTTYPE>::value, "Unmatched const type qualifier!");
#ifndef NDEBUG
    assert(dynamic_cast<CASTTYPE*>(&leader_.get()) != nullptr);
#endif
    return static_cast<CASTTYPE&>(leader_.get());
  }

  template<typename CASTTYPE>
  CASTTYPE& getCastedElement(size_t i) const
  {
    static_assert(std::is_const<T>::value == std::is_const<CASTTYPE>::value, "Unmatched const type qualifier!");
#ifndef NDEBUG
    assert(dynamic_cast<CASTTYPE*>(&(*this)[i]) != nullptr);
#endif
    return static_cast<CASTTYPE&>((*this)[i]);
  }

private:
  std::reference_wrapper<T> leader_;
};

class TWF
{
public:
  static void mw_accept_rejectMove(const RefVectorWithLeader<TWF>& wf_list)
  {
    auto& wf_leader = wf_list.getLeader();
    const int vec_size = wf_list.size();
    int check_size[2];
    const RefVectorWithLeader<TWF>* check_addr[2];
    std::cout << "vec size outside " << vec_size << " addr " << &wf_list << std::endl;
    PRAGMA_OMP_TASKLOOP("omp taskloop default(shared) if(wf_leader.use_tasking)")
    for(int i=0; i<2; i++)
    {
      check_size[i] = wf_list.size();
      check_addr[i] = &wf_list;
    }

    for(int i=0; i<2; i++)
    {
      std::cout << "vec size inside " << wf_list.size() << " addr " << &wf_list << std::endl;
      REQUIRE(check_size[i] == wf_list.size());
    }
  }

private:
  bool use_tasking = false;
};

TEST_CASE("task_loop", "[openmp]")
{
  std::vector<TWF> twf(2);
  std::vector<std::reference_wrapper<TWF>> twf_ref;
  twf_ref.push_back(twf[0]);
  twf_ref.push_back(twf[1]);
  RefVectorWithLeader<TWF> twf_crowd(twf[0], twf_ref);
  TWF::mw_accept_rejectMove(twf_crowd);
}

} // namespace qmcplusplus
