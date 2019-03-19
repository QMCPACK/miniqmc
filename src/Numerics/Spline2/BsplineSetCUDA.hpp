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

#ifndef QMCPLUSPLUS_BSPLINE_SET_CUDA_HPP
#define QMCPLUSPLUS_BSPLINE_SET_CUDA_HPP

#include <cassert>
#include <algorithm>
#include "Devices.h"
#include "Numerics/Spline2/bspline_traits.hpp"
#include "Numerics/Spline2/BsplineAllocatorCUDA.hpp"
#include "Numerics/Containers.h"

namespace qmcplusplus
{
template<Devices DT, typename T>
class BsplineSetCreator;

template<typename T>
struct BsplineSetCreator<Devices::CUDA, T>
{
  using spline_type = typename bspline_traits<Devices::CUDA, T, 3>::SplineType;
  BsplineSetCreator(einspline::Allocator<Devices::CUDA>& allocator,
                    aligned_vector<spline_type*>& minded_splines)
      : allocator_(allocator), minded_splines_(minded_splines){};

  template<typename HOST_SPLINE>
  void operator()(aligned_vector < HOST_SPLINE*> & cpu_splines, int device_block )
  {
    using HT = typename bspline_type<HOST_SPLINE>::value_type;
    T dummy_DT;
    HT dummy_T;
    allocator_.createMultiBspline(cpu_splines, minded_splines_[device_block], dummy_T, dummy_DT);
  }

  template<typename HOST_SPLINE>
  void operator()(HOST_SPLINE*& cpu_spline, int device_block )
  {
    using HT = typename bspline_type<HOST_SPLINE>::value_type;
    T dummy_DT;
    HT dummy_T;
    allocator_.createMultiBspline(cpu_spline, minded_splines_[device_block], dummy_T, dummy_DT);
  }

private:
  einspline::Allocator<Devices::CUDA>& allocator_;
  aligned_vector<spline_type*>& minded_splines_;
};


} // namespace qmcplusplus

#endif
