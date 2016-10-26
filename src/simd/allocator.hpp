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

#if (__cplusplus >= 201103L)

#include <config.h>
#include <vector>

#if defined(__INTEL_COMPILER)
 #include <tbb/cache_aligned_allocator.h>
#else
 #if defined(HAVE_LIBBOOST)
   #include <boost/align/aligned_allocator.hpp>
 #endif
#endif

namespace qmcplusplus
{

#if  defined(__INTEL_COMPILER)
  template<class T>
    using aligned_allocator=tbb::cache_aligned_allocator<T>;
#else
 #if defined(HAVE_LIBBOOST)
  template<class T>
    using aligned_allocator=boost::alignment::aligned_allocator<T, QMC_CLINE>;
 #else
  template<class T>
    using aligned_allocator=std::allocator<T>;
 #endif
#endif
   template<class T> 
     using aligned_vector = std::vector<T,aligned_allocator<T> >;

   template<typename T>
   inline int getAlignedSize(int n)
   {
     constexpr int ND=QMC_CLINE/sizeof(T);
     return ((n+ND-1)/ND)*ND;
   }

}
#endif

#endif