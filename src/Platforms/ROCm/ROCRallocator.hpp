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
// -*- C++ -*-
/** @file ROCRallocator.hpp
 * this file provides three C++ memory allocators using ROCR specific memory allocation functions.
 *
 * ROCRManagedAllocator allocates ROCR unified memory
 * ROCRAllocator allocates ROCR device memory
 * ROCRHostAllocator allocates ROCR host pinned memory
 */
#ifndef QMCPLUSPLUS_ROCR_ALLOCATOR_H
#define QMCPLUSPLUS_ROCR_ALLOCATOR_H

#include <cstdlib>
#include <stdexcept>
#include <hsa.h>
#include "config.h"
#include "rocrError.h"
#include "CPU/SIMD/allocator_traits.hpp"
#include "CPU/SIMD/alignment.config.h"

namespace qmcplusplus
{
/** allocator locks memory pages allocated by ULPHA
 * @tparam T data type
 * @tparam ULPHA host memory allocator using unlocked page
 *
 * ULPHA cannot be ROCRHostAllocator
 */
template<typename T, class ULPHA = std::allocator<T>>
struct ROCRLockedPageAllocator : public ULPHA
{
  using value_type    = typename ULPHA::value_type;
  using size_type     = typename ULPHA::size_type;
  using pointer       = typename ULPHA::pointer;
  using const_pointer = typename ULPHA::const_pointer;

  ROCRLockedPageAllocator() = default;
  template<class U, class V>
  ROCRLockedPageAllocator(const ROCRLockedPageAllocator<U, V>&)
  {}
  template<class U, class V>
  struct rebind
  {
    typedef ROCRLockedPageAllocator<U, V> other;
  };

  value_type* allocate(std::size_t n)
  {
    static_assert(std::is_same<T, value_type>::value, "ROCRLockedPageAllocator and ULPHA data types must agree!");
    value_type* pt = ULPHA::allocate(n);
    rocrErrorCheck(hsa_memory_register(pt, n * sizeof(T)), "hsa_memory_register failed in ROCRLockedPageAllocator!");
    return pt;
  }

  void deallocate(value_type* pt, std::size_t n)
  {
    rocrErrorCheck(hsa_memory_deregister(pt, n * sizeof(T)), "hsa_memory_deregister failed in ROCRLockedPageAllocator!");
    ULPHA::deallocate(pt, n);
  }
};

} // namespace qmcplusplus

#endif
