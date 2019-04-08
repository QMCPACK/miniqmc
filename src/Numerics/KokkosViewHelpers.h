////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2017 QMCPACK developers.
//
// File developed by: L. Shulenburger
//
// File created by: L. Shulenburger
////////////////////////////////////////////////////////////////////////////////
// -*- C++ -*-

/**
 * @file KokkosViewHelpers.h
 * @brief utility routines to do linear algebra with Kokkos
 */

#ifndef QMCPACK_KOKKOSVIEWHELPERS_H
#define QMCPACK_KOKKOSVIEWHELPERS_H
#include <Kokkos_Core.hpp>
#include <impl/Kokkos_Timer.hpp>

namespace qmcplusplus 
{

// assuming both views have the same dimensionality and 
// that the types they hold can assigned between
// CURRENTLY ONLY IMPLEMENTED UP TO RANK 5 tensors

// in many places you can use Kokkos::deep-copy instead, but these will work
// if there is a type conversion between compatible types in source and
// destination, or if source and destination have different layouts

template<class ViewType1, class ViewType2>
void elementWiseCopy(ViewType1 destination, ViewType2 source, 
		     typename std::enable_if<ViewType1::rank==1>::type* = 0,
		     typename std::enable_if<ViewType2::rank==1>::type* = 0) {
  for (int i = 0; i < ViewType1::rank; i++) {
    assert(destination.extent(i) == source.extent(i));
  }
  Kokkos::parallel_for("elementWiseCopy::copy_elements_rk1", 
		       destination.extent(0),
		       KOKKOS_LAMBDA(const int& i0) {
			 destination(i0) = source(i0);
		       });
  Kokkos::fence();
}
template<class ViewType1, class ViewType2>
void elementWiseCopy(ViewType1 destination, ViewType2 source, 
		     typename std::enable_if<ViewType1::rank==2>::type* = 0,
		     typename std::enable_if<ViewType2::rank==2>::type* = 0) {
  for (int i = 0; i < ViewType1::rank; i++) {
    assert(destination.extent(i) == source.extent(i));
  }
  Kokkos::parallel_for("elementWiseCopy::copy_elements_rk2", 
		       Kokkos::MDRangePolicy<Kokkos::Rank<2,Kokkos::Iterate::Left> >({0,0}, {destination.extent(0),destination.extent(1)}), 
		       KOKKOS_LAMBDA(const int& i0, const int& i1) {
			 destination(i0, i1) = source(i0, i1);
		       });
  Kokkos::fence();
}
 template<class ViewType1, class ViewType2>
void elementWiseCopyTrans(ViewType1 destination, ViewType2 source, 
			  typename std::enable_if<ViewType1::rank==2>::type* = 0,
			  typename std::enable_if<ViewType2::rank==2>::type* = 0) {
   assert(destination.extent(0) == source.extent(1));
   assert(destination.extent(1) == source.extent(0));

   Kokkos::parallel_for("elementWiseCopy::copy_elements_rk2", 
		       Kokkos::MDRangePolicy<Kokkos::Rank<2,Kokkos::Iterate::Left> >({0,0}, {destination.extent(0),destination.extent(1)}), 
		       KOKKOS_LAMBDA(const int& i0, const int& i1) {
			 destination(i0, i1) = source(i1, i0);
		       });
   Kokkos::fence();
}
template<class ViewType1, class ViewType2>
void elementWiseCopy(ViewType1 destination, ViewType2 source, 
		     typename std::enable_if<ViewType1::rank==3>::type* = 0,
		     typename std::enable_if<ViewType2::rank==3>::type* = 0) {
  for (int i = 0; i < ViewType1::rank; i++) {
    assert(destination.extent(i) == source.extent(i));
  }
  Kokkos::parallel_for("elementWiseCopy::copy_elements_rk3", 
			 Kokkos::MDRangePolicy<Kokkos::Rank<3,Kokkos::Iterate::Left> >({0,0,0}, {destination.extent(0),destination.extent(1),destination.extent(2)}), 
			 KOKKOS_LAMBDA(const int& i0, const int& i1, const int& i2) {
			   destination(i0, i1, i2) = source(i0, i1, i2);
			 });
  Kokkos::fence();
}
template<class ViewType1, class ViewType2>
void elementWiseCopy(ViewType1 destination, ViewType2 source, 
		     typename std::enable_if<ViewType1::rank==4>::type* = 0,
		     typename std::enable_if<ViewType2::rank==4>::type* = 0) {
  for (int i = 0; i < ViewType1::rank; i++) {
    assert(destination.extent(i) == source.extent(i));
  }
  Kokkos::parallel_for("elementWiseCopy::copy_elements_rk4", 
		       Kokkos::MDRangePolicy<Kokkos::Rank<4,Kokkos::Iterate::Left> >({0,0,0,0}, {destination.extent(0),destination.extent(1),destination.extent(2),destination.extent(3)}), 
		       KOKKOS_LAMBDA(const int& i0, const int& i1, const int& i2, const int& i3) {
			 destination(i0, i1, i2, i3) = source(i0, i1, i2, i3);
		       });
  Kokkos::fence();
}
template<class ViewType1, class ViewType2>
void elementWiseCopy(ViewType1 destination, ViewType2 source, 
		     typename std::enable_if<ViewType1::rank==5>::type* = 0,
		     typename std::enable_if<ViewType2::rank==5>::type* = 0) {
  for (int i = 0; i < ViewType1::rank; i++) {
    assert(destination.extent(i) == source.extent(i));
  }
  Kokkos::parallel_for("elementWiseCopy::copy_elements_rk5", 
		       Kokkos::MDRangePolicy<Kokkos::Rank<5,Kokkos::Iterate::Left> >({0,0,0,0,0}, {destination.extent(0),destination.extent(1),destination.extent(2),destination.extent(3),destination.extent(4)}), 
		       KOKKOS_LAMBDA(const int& i0, const int& i1, const int& i2, const int& i3, const int& i4) {
			 destination(i0, i1, i2, i3, i4) = source(i0, i1, i2, i3, i4);
		       });
  Kokkos::fence();
}

}


#endif
