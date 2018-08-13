////////////////////////////////////////////////////////////////////////////////
//// This file is distributed under the University of Illinois/NCSA Open Source
//// License.  See LICENSE file in top directory for details.
////
//// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
////
//// File developed by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory.
////
//// File created by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory.
//////////////////////////////////////////////////////////////////////////////////

#ifndef QMCPLUSPLUS_OMP_MATRIX_HPP
#define QMCPLUSPLUS_OMP_MATRIX_HPP

#include <OMP_target_test/OMPVector.h>

namespace qmcplusplus
{

template<typename T, class Container = std::vector<T>>
class OMPMatrix: protected OMPVector<T, Container>
{
  private:
  using Base_t = OMPVector<T, Container>;

  protected:
  size_t D1, D2;
  size_t TotSize;
  using Base_t::vec_ptr;
  using Base_t::device_id;

  public:
  inline OMPMatrix(size_t nrow = 0, size_t ncol = 0 , size_t id = 0)
    : D1(nrow), D2(ncol), TotSize(nrow*ncol), Base_t(nrow*ncol, id)
  { }

  inline void resize(size_t nrow, size_t ncol)
  {
    D1 = nrow;
    D2 = ncol;
    TotSize = nrow*ncol;
    Base_t::resize(TotSize);
  }

  // expose base class routines
  using Base_t::update_to_device;
  using Base_t::update_from_device;
  using Base_t::data;
  using Base_t::size;

  inline void update_row_to_device(size_t row) const
  {
#ifdef ENABLE_OFFLOAD
    #pragma omp target update to(vec_ptr[row*D2:D2]) device(device_id)
#endif
  }

  inline void update_row_from_device(size_t row) const
  {
#ifdef ENABLE_OFFLOAD
    #pragma omp target update from(vec_ptr[row*D2:D2]) device(device_id)
#endif
  }

  /// returns a const pointer of i-th row
  inline const T* operator[](size_t i) const { return data() + i*D2; }

  /// returns a pointer of i-th row, g++ iterator problem
  inline T* operator[](size_t i) { return data() + i*D2; }

  // returns the i-th value in D1*D2 vector
  inline T& operator()(size_t i) { return *(data()+i); }
  inline const T& operator()(size_t i) const { return *(data()+i); }

  // returns val(i,j)
  inline T& operator()(size_t i, size_t j) { return *(data() + i*D2 + j); }
  inline const T& operator()(size_t i, size_t j) const { return *(data() + i*D2 + j); }

};

}
#endif
