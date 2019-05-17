////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by:
//
// File created by:
// Jeongnim Kim, jeongnim.kim@intel.com,
//    Intel Corp.
////////////////////////////////////////////////////////////////////////////////
// -*- C++ -*-
/** @file Containers.h
 *
 * Master header to support SoA containers
 */
#ifndef QMCPLUSPLUS_SOA_CONTAINER_H
#define QMCPLUSPLUS_SOA_CONTAINER_H
#include <config.h>
#include <Numerics/OhmmsPETE/TinyVector.h>
#include <Numerics/OhmmsPETE/Tensor.h>
#include <Utilities/SIMD/allocator.hpp>
#include <Numerics/PosTransformer.h>

/* Soa Container for D-dim vectors
 *
 * Alternative to Container<TinyVector<T,D> > to support SoA algorithms
 */
#include <Utilities/SIMD/allocator.hpp>
#include <Utilities/SIMD/algorithm.hpp>

namespace qmcplusplus
{
/** SoA adaptor class for ParticleAttrib<TinyVector<T,D> >
 * @tparm T data type, float, double, complex<float>, complex<double>
 */
template<typename T, unsigned D>
struct VectorSoAContainer
{
  using Type_t    = TinyVector<T, D>;
  using Element_t = T;
  /////alias to ParticleAttrib<T1,D>
  // template <class T1> using PosArray = ParticleAttrib<TinyVector<T1,D> >;
  /// number of elements
  size_t nLocal;
  /// number of elements + padded
  size_t nGhosts;
  /// number of elements allocated by myAlloc
  size_t nAllocated;
  /// pointer: what type????
  T* myData;
  /// allocator
  aligned_allocator<T> myAlloc;
  /// default constructor
  VectorSoAContainer() { setDefaults(); }
  /// destructor
  ~VectorSoAContainer()
  {
    if (nAllocated > 0) myAlloc.deallocate(myData, nAllocated);
  }

  /// default copy constructor
  VectorSoAContainer(const VectorSoAContainer& in)
  {
    setDefaults();
    resize(in.nLocal);
    simd::copy_n(in.myData, nGhosts * D, myData);
  }

  /// default copy operator
  VectorSoAContainer& operator=(const VectorSoAContainer& in)
  {
    if (myData != in.myData)
    {
      resize(in.nLocal);
      simd::copy_n(in.myData, nGhosts * D, myData);
    }
    return *this;
  }

  /// move constructor
  VectorSoAContainer(VectorSoAContainer&& in) : nLocal(in.nLocal), nGhosts(in.nGhosts)
  {
    nAllocated    = in.nAllocated;
    myData        = in.myData;
    myAlloc       = std::move(in.myAlloc);
    in.myData     = nullptr;
    in.nAllocated = 0;
  }

  /** constructor with size n  without initialization
   */
  explicit VectorSoAContainer(size_t n)
  {
    setDefaults();
    resize(n);
  }

  /** constructor with ParticleAttrib<T1,D> */
  template<typename T1>
  VectorSoAContainer(const ParticleAttrib<TinyVector<T1, D>>& in)
  {
    setDefaults();
    resize(in.size());
    copyIn(in);
  }

  template<typename T1>
  VectorSoAContainer& operator=(const ParticleAttrib<TinyVector<T1, D>>& in)
  {
    if (nLocal != in.size()) resize(in.size());
    copyIn(in);
    return *this;
  }

  /** need A=0.0;
   */
  template<typename T1>
  VectorSoAContainer& operator=(T1 in)
  {
    std::fill(myData, myData + nGhosts * D, static_cast<T>(in));
    return *this;
  }

  /// initialize the data members
  __forceinline void setDefaults()
  {
    nLocal     = 0;
    nGhosts    = 0;
    nAllocated = 0;
    myData     = nullptr;
  }

  /** resize myData
   * @param n nLocal
   *
   * nAllocated is used to ensure no memory leak
   */
  __forceinline void resize(size_t n)
  {
    if (nAllocated) myAlloc.deallocate(myData, nAllocated);
    nLocal     = n;
    nGhosts    = getAlignedSize<T>(n);
    nAllocated = nGhosts * D;
    myData     = myAlloc.allocate(nAllocated);
  }

  /** reset by pre-allocated data
   * @param n new nLocal
   * @param n_padded new nGhosts
   * @param ptr new myData
   *
   * Free existing memory and reset the internal variables
   */
  __forceinline void attachReference(size_t n, size_t n_padded, T* ptr)
  {
    if (nAllocated)
      throw std::runtime_error(
          "Pointer attaching is not allowed on VectorSoAContainer with allocated memory.");
    nAllocated = 0;
    nLocal     = n;
    nGhosts    = n_padded;
    myData     = ptr;
  }

  /// return the physical size
  __forceinline size_t size() const { return nLocal; }
  /// return the physical size
  __forceinline size_t capacity() const { return nGhosts; }

  /** AoS to SoA : copy from ParticleAttrib<>
   *
   * The same sizes are assumed.
   */
  template<typename T1>
  void copyIn(const ParticleAttrib<TinyVector<T1, D>>& in)
  {
    // if(nLocal!=in.size()) resize(in.size());
    PosAoS2SoA(nLocal, D, reinterpret_cast<const T1*>(in.first_address()), D, myData, nGhosts);
  }

  /** SoA to AoS : copy to ParticleAttrib<>
   *
   * The same sizes are assumed.
   */
  template<typename T1>
  void copyOut(ParticleAttrib<TinyVector<T1, D>>& out) const
  {
    PosSoA2AoS(nLocal, D, myData, nGhosts, reinterpret_cast<T1*>(out.first_address()), D);
  }

  /** return TinyVector<T,D>
   */
  __forceinline const Type_t operator[](size_t i) const { return Type_t(myData + i, nGhosts); }

  /// helper class for operator ()(size_t i) to assign a value
  struct Accessor
  {
    size_t M;
    T* _base;
    __forceinline Accessor(T* a, size_t ng) : _base(a), M(ng) {}
    __forceinline Accessor& operator=(const TinyVector<T, D>& rhs)
    {
#pragma unroll
      for (size_t i = 0; i < D; ++i)
        *(_base + M * i) = rhs[i];
      return *this;
    }

    /** asign value */
    template<typename T1>
    __forceinline Accessor& operator=(T1 rhs)
    {
#pragma unroll
      for (size_t i = 0; i < D; ++i)
        *(_base + M * i) = rhs;
      return *this;
    }
  };

  /** access operator for assignment of the i-th value
   *
   * Use for (*this)[i]=TinyVector<T,D>;
   */
  __forceinline Accessor operator()(size_t i) { return Accessor(myData + i, nGhosts); }
  /// return the base
  __forceinline T* data() { return myData; }
  /// return the base
  __forceinline const T* data() const { return myData; }
  /// return the pointer of the i-th components
  __forceinline T* restrict data(size_t i) { return myData + i * nGhosts; }
  /// return the const pointer of the i-th components
  __forceinline const T* restrict data(size_t i) const { return myData + i * nGhosts; }

  /** serialization function */
  template<class Archive>
  void serialize(Archive& ar, const unsigned int version)
  {
    // ar & m_data;
    ar& nLocal& nGhosts& myData;
  }
};

// Incorrect: provide wrapper class
// BOOST_CLASS_TRACKING(Pos3DSoA<double,3>, boost::serialization::track_never)
// BOOST_CLASS_TRACKING(Pos3DSoA<float,3>, boost::serialization::track_never)
} // namespace qmcplusplus
#endif
