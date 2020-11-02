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
/** @file SYCLallocator.hpp
 * this file provides three C++ memory allocators using SYCL specific memory allocation functions.
 *
 * SYCLManagedAllocator allocates SYCL unified memory
 * SYCLAllocator allocates SYCL device memory
 * SYCLHostAllocator allocates SYCL host pinned memory
 */
#ifndef QMCPLUSPLUS_SYCL_ALLOCATOR_H
#define QMCPLUSPLUS_SYCL_ALLOCATOR_H

#include <cstdlib>
#include <stdexcept>
#include <CL/sycl.hpp>
#include "config.h"
#include "CPU/SIMD/allocator_traits.hpp"
#include "CPU/SIMD/alignment.config.h"

namespace qmcplusplus
{
/** allocator for SYCL unified memory
 * @tparam T data type
 */
template<typename T>
struct SYCLManagedAllocator
{
  typedef T value_type;
  typedef size_t size_type;
  typedef T* pointer;
  typedef const T* const_pointer;

  SYCLManagedAllocator() = default;
  template<class U>
  SYCLManagedAllocator(const SYCLManagedAllocator<U>&)
  {}

  template<class U>
  struct rebind
  {
    typedef SYCLManagedAllocator<U> other;
  };

  T* allocate(std::size_t n)
  {
    sycl::queue queue;
    void* pt = sycl::aligned_alloc_shared(QMC_CLINE, sizeof(int) * n, queue);
    return static_cast<T*>(pt);
  }

  void deallocate(T* p, std::size_t)
  {
    sycl::queue queue;
    sycl::free(p, queue);
  }
};

template<class T1, class T2>
bool operator==(const SYCLManagedAllocator<T1>&, const SYCLManagedAllocator<T2>&)
{
  return true;
}
template<class T1, class T2>
bool operator!=(const SYCLManagedAllocator<T1>&, const SYCLManagedAllocator<T2>&)
{
  return false;
}

/** allocator for SYCL device memory
 * @tparam T data type
 */
template<typename T>
struct SYCLAllocator
{
  typedef T value_type;
  typedef size_t size_type;
  typedef T* pointer;
  typedef const T* const_pointer;

  SYCLAllocator() = default;
  template<class U>
  SYCLAllocator(const SYCLAllocator<U>&)
  {}

  template<class U>
  struct rebind
  {
    typedef SYCLAllocator<U> other;
  };

  T* allocate(std::size_t n)
  {
    sycl::queue queue;
    void* pt = sycl::aligned_alloc(QMC_CLINE, sizeof(int) * n, queue, sycl::usm::alloc::device);
    return static_cast<T*>(pt);
  }

  void deallocate(T* p, std::size_t)
  {
    sycl::queue queue;
    sycl::free(p, queue);
  }
};

template<class T1, class T2>
bool operator==(const SYCLAllocator<T1>&, const SYCLAllocator<T2>&)
{
  return true;
}
template<class T1, class T2>
bool operator!=(const SYCLAllocator<T1>&, const SYCLAllocator<T2>&)
{
  return false;
}

template<typename T>
struct allocator_traits<SYCLAllocator<T>>
{
  const static bool is_host_accessible = false;
};

/** allocator for SYCL host pinned memory
 * @tparam T data type
 */
template<typename T>
struct SYCLHostAllocator
{
  typedef T value_type;
  typedef size_t size_type;
  typedef T* pointer;
  typedef const T* const_pointer;

  SYCLHostAllocator() = default;
  template<class U>
  SYCLHostAllocator(const SYCLHostAllocator<U>&)
  {}

  template<class U>
  struct rebind
  {
    typedef SYCLHostAllocator<U> other;
  };

  T* allocate(std::size_t n)
  {
    sycl::queue queue;
    void* pt = sycl::aligned_alloc_host(QMC_CLINE, sizeof(int) * n, queue);
    return static_cast<T*>(pt);
  }

  void deallocate(T* p, std::size_t)
  {
    sycl::queue queue;
    sycl::free(p, queue);
  }
};

template<class T1, class T2>
bool operator==(const SYCLHostAllocator<T1>&, const SYCLHostAllocator<T2>&)
{
  return true;
}
template<class T1, class T2>
bool operator!=(const SYCLHostAllocator<T1>&, const SYCLHostAllocator<T2>&)
{
  return false;
}

} // namespace qmcplusplus

#endif
