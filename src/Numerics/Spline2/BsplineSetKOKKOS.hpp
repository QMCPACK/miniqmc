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

#ifndef QMCPLUSPLUS_BSPLINE_SET_KOKKOS_HPP
#define QMCPLUSPLUS_BSPLINE_SET_KOKKOS_HPP

#include <cassert>
#include <algorithm>
#include "Devices.h"
#include "Numerics/Spline2/bspline_traits.hpp"
#include "Numerics/Spline2/BsplineAllocatorKOKKOS.hpp"
#include "Numerics/Containers.h"

namespace qmcplusplus
{
template<Devices DT, typename T>
class BsplineSetCreator;

/** The kokkos Bspline set is in a view instead of an aligned vector
 *  I think this adds no value here
 */
template<typename T>
struct BsplineSetCreator<Devices::KOKKOS, T>
{
  static constexpr Devices DT = Devices::KOKKOS;
  using spline_type = typename bspline_traits<DT, T, 3>::SplineType;
  BsplineSetCreator(einspline::Allocator<DT>& allocator, Kokkos::View<spline_type*>& minded_splines) : allocator_(allocator), minded_splines_(minded_splines) {};
  void operator()(int block, TinyVector<T,3> start, TinyVector<T,3> end, int nx, int ny, int nz, int splines_per_block)
  {
    TinyVector<int, 3> ng(nx, ny, nz);
    spline_type* pspline_temp;
    allocator_.createMultiBspline(pspline_temp, T(0), start, end, ng, PERIODIC,
                                  splines_per_block);
    minded_splines_(block) = *pspline_temp;
  };
private:
  einspline::Allocator<DT>& allocator_;
  Kokkos::View<spline_type*>& minded_splines_;
};

template<Devices DT, typename T>
class BsplineSet;

/** This is specialized completely for KOKKOS since it doesn't make an
 *  aligned array of spline_type* but a Kokkos::View<spline_type*>,
 *  (which is actually a one dimensional array of spline_type),
 *  which to me looks like a non interoperable vector. With reference
 *  counting better left to the application developers discretion.
 *  As a result I'm still having this use a creator with the hope that
 *  The following specialization can be factored away.
 */
template<typename T>
class BsplineSet<Devices::KOKKOS,T>
{
  static constexpr Devices DT = Devices::KOKKOS;
public:
  using spline_type = typename bspline_traits<DT, T, 3>::SplineType;
  BsplineSet(int num_blocks)
  {
    //assert(num_blocks > minded_splines_.size());
    Kokkos::resize(minded_splines_, num_blocks);
  }

  ~BsplineSet()
  {
    minded_splines_ = Kokkos::View<spline_type*>();
  }

  void resize(int size)
  {
    assert(size > minded_splines_.size());
    Kokkos::resize(minded_splines_, size); 
  }
  
    //aligned_vector<spline_type*>& get() { return minded_splines_; }

  spline_type* operator[] (int block_index) { return &minded_splines_(block_index); }

  BsplineSetCreator<DT, T> creator() { return BsplineSetCreator<DT,T>(allocator_, minded_splines_); }

  /** This signature is different from CPU version. */
  void setCoefficientsForOneOrbital(int spline_index, Kokkos::View<T***>& coeff, int block)
  {
      allocator_.setCoefficientsForOneOrbital(spline_index,
					      coeff,
					      &(minded_splines_(block)));
  }

protected:
  einspline::Allocator<DT> allocator_;
  Kokkos::View<spline_type*> minded_splines_;
};

}

#endif
