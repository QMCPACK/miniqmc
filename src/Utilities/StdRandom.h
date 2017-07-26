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

#ifndef QMCPLUSPLUS_CPP11_STDRANDOM_H
#define QMCPLUSPLUS_CPP11_STDRANDOM_H

#include <iterator>
#include <random>
#include "Message/OpenMP.h"

template<typename T, typename RNG=std::mt19937>
struct StdRandom
{
  /// real result type
  typedef T result_type;
  /// randmon number generator [0,max) where max depends on the generator type
  typedef RNG generator_type;
  /// unsigned integer type
  typedef uint32_t uint_type; //typename generator_type::result_type uint_type;
  /// real random generator [0,1)
  typedef std::uniform_real_distribution<T> uniform_distribution_type;
  /// normal real random generator [0,1)
  typedef std::normal_distribution<T> normal_distribution_type;
  ///number of contexts
  int nContexts;
  ///context number
  int myContext;
  ///offset of the random seed
  int baseOffset;
  ///random number generator
  RNG myRNG;
  /// uniform generator
  uniform_distribution_type uniform;
  /// normal generator
  normal_distribution_type normal;

  StdRandom()
    : nContexts(1),myContext(0),baseOffset(0), uniform(T(0),T(1)), normal(T(0),T(1))
  {
    myRNG.seed(MakeSeed(omp_get_thread_num(),omp_get_num_threads()));
  }

  explicit StdRandom(uint_type iseed): 
    nContexts(1),myContext(0),baseOffset(0)
    //, uniform(T(0),T(1)), normal(T(0),T(1))
  {
    if(iseed==0) iseed=MakeSeed(0,1);
    myRNG.seed(iseed);
  }

  /** copy constructor
   */
  template<typename T1> 
  StdRandom(const StdRandom<T1,RNG>& rng):
      nContexts(1),myContext(0),baseOffset(0),
      myRNG(rng.myRng),uniform(T(0),T(1)),
      normal(T(0),T(1)) { }

  /** initialize the stream */
  inline void init(int i, int nstr, int iseed_in, uint_type offset=1)
  {
    uint_type baseSeed=iseed_in;
    myContext=i;
    nContexts=nstr;
    if(iseed_in<=0)
      baseSeed=MakeSeed(i,nstr);
    baseOffset=offset;
    myRNG.seed(baseSeed);
  }

  template<typename T1>
  inline void reset(const StdRandom<T1,RNG>& rng)
  {
    myRNG=rng; //copy the state
  }

  ///get baseOffset
  inline int offset() const
  {
    return baseOffset;
  }
  ///assign baseOffset
  inline int& offset()
  {
    return baseOffset;
  }

  ///assign seed
  inline void seed(uint_type aseed)
  {
    myRNG.seed(aseed);
  }

  /** return a random number [0,1)
  */
  inline result_type rand()
  {
    return uniform(myRNG);
  }
  /** return a random number [0,1)
  */
  inline result_type operator()()
  {
    return uniform(myRNG);
  }

  /** generate a series of random numbers */
  inline void generate_uniform(T* restrict d, int n)
  {
    for(int i=0; i<n; ++i) d[i]=uniform(myRNG);
  }

  inline void generate_normal(T* restrict d, int n)
  {
    BoxMuller2::generate(*this,d,n);
  }

  /** return a random integer
  */
  inline uint32_t irand()
  {
    std::uniform_int_distribution<uint32_t> a;
    return a(myRNG);
  }
};
#endif
