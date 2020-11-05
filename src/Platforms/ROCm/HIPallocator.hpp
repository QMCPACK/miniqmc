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
/** @file HIPallocator.hpp
 * this file provides three C++ memory allocators using HIP specific memory allocation functions.
 *
 * HIPManagedAllocator allocates HIP unified memory
 * HIPAllocator allocates HIP device memory
 * HIPHostAllocator allocates HIP host pinned memory
 */
#ifndef QMCPLUSPLUS_HIP_ALLOCATOR_H
#define QMCPLUSPLUS_HIP_ALLOCATOR_H

#include <cstdlib>
#include <stdexcept>
#include <hip/hip_runtime_api.h>
#include "config.h"
#include "hipError.h"
#include "CPU/SIMD/allocator_traits.hpp"
#include "CPU/SIMD/alignment.config.h"

namespace qmcplusplus
{
/** allocator for HIP unified memory
 * @tparam T data type
 */
/*
template<typename T>
struct HIPManagedAllocator
{
  typedef T value_type;
  typedef size_t size_type;
  typedef T* pointer;
  typedef const T* const_pointer;

  HIPManagedAllocator() = default;
  template<class U>
  HIPManagedAllocator(const HIPManagedAllocator<U>&)
  {}

  template<class U>
  struct rebind
  {
    typedef HIPManagedAllocator<U> other;
  };

  T* allocate(std::size_t n)
  {
    void* pt;
    hipErrorCheck(hipMallocManaged(&pt, n * sizeof(T)), "Allocation failed in HIPManagedAllocator!");
    if ((size_t(pt)) & (QMC_CLINE - 1))
      throw std::runtime_error("Unaligned memory allocated in HIPManagedAllocator");
    return static_cast<T*>(pt);
  }
  void deallocate(T* p, std::size_t) { hipErrorCheck(hipFree(p), "Deallocation failed in HIPManagedAllocator!"); }
};

template<class T1, class T2>
bool operator==(const HIPManagedAllocator<T1>&, const HIPManagedAllocator<T2>&)
{
  return true;
}
template<class T1, class T2>
bool operator!=(const HIPManagedAllocator<T1>&, const HIPManagedAllocator<T2>&)
{
  return false;
}
*/

/** allocator for HIP device memory
 * @tparam T data type
 */
template<typename T>
struct HIPAllocator
{
  typedef T value_type;
  typedef size_t size_type;
  typedef T* pointer;
  typedef const T* const_pointer;

  HIPAllocator() = default;
  template<class U>
  HIPAllocator(const HIPAllocator<U>&)
  {}

  template<class U>
  struct rebind
  {
    typedef HIPAllocator<U> other;
  };

  T* allocate(std::size_t n)
  {
    void* pt;
    hipErrorCheck(hipMalloc(&pt, n * sizeof(T)), "Allocation failed in HIPAllocator!");
    return static_cast<T*>(pt);
  }
  void deallocate(T* p, std::size_t) { hipErrorCheck(hipFree(p), "Deallocation failed in HIPAllocator!"); }
};

template<class T1, class T2>
bool operator==(const HIPAllocator<T1>&, const HIPAllocator<T2>&)
{
  return true;
}
template<class T1, class T2>
bool operator!=(const HIPAllocator<T1>&, const HIPAllocator<T2>&)
{
  return false;
}

template<typename T>
bool isHIPPtrDevice(const T* ptr)
{
  hipPointerAttribute_t attr;
  hipErrorCheck(hipPointerGetAttributes(&attr, ptr), "hipPointerGetAttributes failed!");
  return attr.memoryType == hipMemoryTypeDevice;
}

template<typename T>
struct allocator_traits<HIPAllocator<T>>
{
  const static bool is_host_accessible = false;
};

/** allocator for HIP host pinned memory
 * @tparam T data type
 */
template<typename T>
struct HIPHostAllocator
{
  typedef T value_type;
  typedef size_t size_type;
  typedef T* pointer;
  typedef const T* const_pointer;

  HIPHostAllocator() = default;
  template<class U>
  HIPHostAllocator(const HIPHostAllocator<U>&)
  {}

  template<class U>
  struct rebind
  {
    typedef HIPHostAllocator<U> other;
  };

  T* allocate(std::size_t n)
  {
    void* pt;
    hipErrorCheck(hipHostMalloc(&pt, n * sizeof(T)), "Allocation failed in HIPHostAllocator!");
    return static_cast<T*>(pt);
  }
  void deallocate(T* p, std::size_t) { hipErrorCheck(hipFreeHost(p), "Deallocation failed in HIPHostAllocator!"); }
};

template<class T1, class T2>
bool operator==(const HIPHostAllocator<T1>&, const HIPHostAllocator<T2>&)
{
  return true;
}
template<class T1, class T2>
bool operator!=(const HIPHostAllocator<T1>&, const HIPHostAllocator<T2>&)
{
  return false;
}

/** allocator locks memory pages allocated by ULPHA
 * @tparam T data type
 * @tparam ULPHA host memory allocator using unlocked page
 *
 * ULPHA cannot be HIPHostAllocator
 */
template<typename T, class ULPHA = std::allocator<T>>
struct HIPLockedPageAllocator : public ULPHA
{
  using value_type    = typename ULPHA::value_type;
  using size_type     = typename ULPHA::size_type;
  using pointer       = typename ULPHA::pointer;
  using const_pointer = typename ULPHA::const_pointer;

  HIPLockedPageAllocator() = default;
  template<class U, class V>
  HIPLockedPageAllocator(const HIPLockedPageAllocator<U, V>&)
  {}
  template<class U, class V>
  struct rebind
  {
    typedef HIPLockedPageAllocator<U, V> other;
  };

  value_type* allocate(std::size_t n)
  {
    static_assert(std::is_same<T, value_type>::value, "HIPLockedPageAllocator and ULPHA data types must agree!");
    value_type* pt = ULPHA::allocate(n);
    hipErrorCheck(hipHostRegister(pt, n*sizeof(T), hipHostRegisterDefault), "hipHostRegister failed in HIPLockedPageAllocator!");
    return pt;
  }

  void deallocate(value_type* pt, std::size_t n)
  {
    hipErrorCheck(hipHostUnregister(pt), "hipHostUnregister failed in HIPLockedPageAllocator!");
    ULPHA::deallocate(pt, n);
  }
};

} // namespace qmcplusplus

#endif
