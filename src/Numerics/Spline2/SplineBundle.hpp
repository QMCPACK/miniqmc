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
// -*- C++ -*-

#ifndef QMCPLUSPLUS_SPLINE_BUNDLE_HPP
#define QMCPLUSPLUS_SPLINE_BUNDLE_HPP

#include "Numerics/Spline2/bspline_traits.hpp"
#include "Utilities/SIMD/allocator.hpp"
/** template class to wrap einsplines held by SPO's
 *  allowing their life spans to be handled through a shared_ptr
 *  We'll see if there is any really performance hit for this
 */
namespace qmcplusplus
{

template<Devices DT, typename T>
struct SplineBundle
{
  using spline_type = typename bspline_traits<DT, T, 3>::SplineType;
  aligned_vector<spline_type*> einsplines;
public:
  void resize(size_t size) { einsplines.resize(size); }
  spline_type* operator[](size_t index) { return einsplines[index]; };
  ~SplineBundle() { }
};

#ifdef QMC_USE_CUDA
  /** Purposely does not contain a vector known as eisplines
   */
template<typename T>
struct SplineBundle<Devices::CUDA, T>
{
  using device_spline_type = typename bspline_traits<Devices::CUDA, T, 3>::SplineType;
  using host_spline_type = typename bspline_traits<Devices::CPU, T, 3>::SplineType;
  aligned_vector<device_spline_type*> device_einsplines;
  //These might not be needed after transfer to GPU
  aligned_vector<host_spline_type*> host_einsplines;
};
#endif

}
#endif
