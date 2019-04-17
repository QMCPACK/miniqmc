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

#ifndef QMCPLUSPLUS_BSPLINE_SET_HPP
#define QMCPLUSPLUS_BSPLINE_SET_HPP

#include <cassert>
#include <algorithm>
#include "Devices.h"
#include "Numerics/Spline2/bspline_traits.hpp"
#include "Numerics/Spline2/BsplineAllocator.hpp"
#include "Numerics/Containers.h"

// This is needed here so BsplineSetCreator functor is defined
#ifdef QMC_USE_CUDA
#include "Numerics/Spline2/BsplineSetCUDA.hpp"
#endif
#ifdef QMC_USE_KOKKOS
#include "Numerics/Spline2/BsplineSetKOKKOS.hpp"
#endif

namespace qmcplusplus
{

template<Devices DT, typename T>
struct BsplineSetCreator
{
  using spline_type = typename bspline_traits<DT, T, 3>::SplineType;
  BsplineSetCreator(einspline::Allocator<DT>& allocator, aligned_vector<spline_type*>& minded_splines) : allocator_(allocator), minded_splines_(minded_splines) {};
  void operator()(int block, TinyVector<T,3> start, TinyVector<T,3> end, int nx, int ny, int nz, int splines_per_block)
  {
    TinyVector<int, 3> ng(nx, ny, nz);
    allocator_.createMultiBspline(minded_splines_[block], T(0), start, end, ng, PERIODIC,
                                  splines_per_block);
  };
private:
  einspline::Allocator<DT>& allocator_;
  aligned_vector<spline_type*>& minded_splines_;
};

/** template class to wrap einsplines held by SPO's
 *  allowing their life spans to be handled through a shared_ptr
 *  and without all the is_owned, copied, etc. flags in EinsplineSPO
 *  So far this has negligible effect on performance
 */
template<Devices DT, typename T>
class BsplineSet
{
public:
  using spline_type = typename bspline_traits<DT, T, 3>::SplineType;
  BsplineSet(int num_blocks)
  {
    assert(num_blocks > minded_splines_.size());
    minded_splines_.resize(num_blocks, nullptr);
  }

  ~BsplineSet()
  {
    std::for_each(minded_splines_.begin(), minded_splines_.end(),
                  [&](spline_type* spline) { this->allocator_.destroy(spline); });
  }

  void resize(int size)
  {
    assert(size > minded_splines_.size());
    minded_splines_.resize(size, nullptr);
  }
  
  aligned_vector<spline_type*>& get() { return minded_splines_; }

  spline_type*& operator[] (int block_index) { return minded_splines_[block_index]; }

  BsplineSetCreator<DT, T> creator() { return BsplineSetCreator<DT,T>(allocator_, minded_splines_); }

  void setCoefficientsForOneOrbital(int spline_index, Array<T, 3>& coeff, int block)
  {
      allocator_.setCoefficientsForOneOrbital(spline_index,
					      coeff,
					      minded_splines_[block]);
  }

protected:
  einspline::Allocator<DT> allocator_;
  aligned_vector<spline_type*> minded_splines_;
};

  extern template class BsplineSet<Devices::CPU, double>;
  extern template class BsplineSet<Devices::CPU, float>;

 #ifdef QMC_USE_CUDA
extern template class BsplineSet<Devices::CUDA, double>;
extern template class BsplineSet<Devices::CUDA, float>;
extern template class BsplineSetCreator<Devices::CUDA, double>;
extern template class BsplineSetCreator<Devices::CUDA, float>;
#endif

#ifdef QMC_USE_KOKKOS
extern template class BsplineSet<Devices::KOKKOS, double>;
extern template class BsplineSet<Devices::KOKKOS, float>;
extern template class BsplineSetCreator<Devices::KOKKOS, double>;
extern template class BsplineSetCreator<Devices::KOKKOS, float>;
#endif    

}


#endif
