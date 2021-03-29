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


#ifndef QMCPLUSPLUS_PINNED_ALLOCATOR_H
#define QMCPLUSPLUS_PINNED_ALLOCATOR_H

#include <memory>
#include "CPU/SIMD/aligned_allocator.hpp"
#if defined(QMC_ENABLE_CUDA)
#include "CUDA/CUDAallocator.hpp"
#elif defined(QMC_ENABLE_ROCM)
#include "ROCm/HIPallocator.hpp"
#elif defined(QMC_ENABLE_ONEAPI)
#include "OneAPI/SYCLallocator.hpp"
#endif

namespace qmcplusplus
{

template<typename T>
#if defined(QMC_ENABLE_CUDA)
using PinnedAllocator = CUDALockedPageAllocator<T>;
#elif defined(QMC_ENABLE_ROCM)
using PinnedAllocator = HIPLockedPageAllocator<T>;
#elif defined(QMC_ENABLE_ONEAPI)
using PinnedAllocator = SYCLHostAllocator<T>;
#else
using PinnedAllocator = std::allocator<T>;
#endif

template<typename T, size_t ALIGN = QMC_SIMD_ALIGNMENT>
#if defined(QMC_ENABLE_CUDA)
using PinnedAlignedAllocator = CUDALockedPageAllocator<T, aligned_allocator<T, ALIGN>>;
#elif defined(QMC_ENABLE_ROCM)
using PinnedAlignedAllocator = HIPLockedPageAllocator<T, aligned_allocator<T, ALIGN>>;
#elif defined(QMC_ENABLE_ONEAPI)
using PinnedAlignedAllocator = SYCLHostAllocator<T>;
#else
using PinnedAlignedAllocator = aligned_allocator<T, ALIGN>;
#endif

} // namespace qmcplusplus

#endif
