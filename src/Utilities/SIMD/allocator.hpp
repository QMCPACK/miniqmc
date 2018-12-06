//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by:
//
// File created by: Jeongnim Kim, jeongnim.kim@intel.com, Intel Corp.
//////////////////////////////////////////////////////////////////////////////////////
// -*- C++ -*-
/** @file allocator.hpp
 */
#ifndef QMCPLUSPLUS_ALLOCATOR_H
#define QMCPLUSPLUS_ALLOCATOR_H

#include <config.h>
#include <vector>
#include <cstdlib>
//#include <boost/align/aligned_allocator.hpp>
#include "Utilities/SIMD/Mallocator.hpp"

namespace qmcplusplus
{
template<class T>
using aligned_allocator = qmcplusplus::Mallocator<T, QMC_CLINE>;
  //using aligned_allocator = boost::alignment::aligned_allocator<T, QMC_CLINE>;
template<class T>
using aligned_vector = std::vector<T, aligned_allocator<T>>;

} // namespace qmcplusplus

template<typename T>
inline size_t getAlignedSize(size_t n)
{
  constexpr size_t ND = QMC_CLINE / sizeof(T);
  return ((n + ND - 1) / ND) * ND;
}

template<typename T>
inline size_t getAlignment()
{
  return QMC_CLINE / sizeof(T);
}

#endif
