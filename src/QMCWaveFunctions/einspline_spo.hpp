////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by:
// Jeongnim Kim, jeongnim.kim@intel.com,
//    Intel Corp.
// Amrita Mathuriya, amrita.mathuriya@intel.com,
//    Intel Corp.
//
// File created by:
// Jeongnim Kim, jeongnim.kim@intel.com,
//    Intel Corp.
// ////////////////////////////////////////////////////////////////////////////////
// -*- C++ -*-
/** @file einspline_spo.hpp
 */
#ifndef QMCPLUSPLUS_EINSPLINE_SPO_HPP
#define QMCPLUSPLUS_EINSPLINE_SPO_HPP
#include <Utilities/Configuration.h>
#include <Utilities/NewTimer.h>
#include <Particle/ParticleSet.h>
#include <Numerics/Spline3/KokkosMultiBspline.h>
#include "QMCWaveFunctions/SPOSet.h"
#include <iostream>

namespace qmcplusplus
{
template<typename T, int blockSize>
struct einspline_spo : public SPOSet
{
  // Eventually need to define a layout that includes the blocking on groups of splines
  // which is the first dimension in this case and the v, g and h containers
  using spline_type     = multi_UBspline<T, blockSize, 3>;
  using vContainer_type = Kokkos::View<T*>;
  using gContainer_type = Kokkos::View<T * [3], Kokkos::LayoutLeft>;
  using hContainer_type = Kokkos::View<T * [6], Kokkos::LayoutLeft>;
  using lattice_type    = CrystalLattice<T, 3>;

  /// number of splines
  int nSplines;
  
  lattice_type Lattice;
  spline_type spline;
  
  vContainer_type psi;
  gContainer_type grad;
  hContainer_type hess;

  /// Timer
  NewTimer* timer;

  /// default constructor
  einspline_spo()
      : nSplines(0)
  {
    timer = TimerManager.createTimer("Single-Particle Orbitals", timer_level_fine);
  }
  /// disable copy constructor
  einspline_spo(const einspline_spo& in) = default;
  /// disable copy operator
  einspline_spo& operator=(const einspline_spo& in) = delete;

  /// destructors
  ~einspline_spo() = default;

  /// resize the containers
  void resize()
  {
    if (nSplines != 0) {
      psi  = vContainer_type("Psi", nSplines);
      grad = gContainer_type("Grad", nSplines);
      hess = hContainer_type("Hess", nSplines);
    }
  }

  // fix for general num_splines
  void set(int nx, int ny, int nz, int num_splines, bool init_random = true)
  { 
    nSplines         = num_splines;
    resize();
    std::vector<int> ng{nx, ny, nz};
    std::vector<double> start(3,0);
    std::vector<double> end(3,1);

    spline.initialize(ng, start, end, 0, nSplines);
    RandomGenerator<T> myrandom(11);
 
    const int totalNumCoefs = spline.coef.extent(0)*spline.coef.extent(1)*spline.coef.extent(2);
    //T* mydata = new T[totalNumCoefs];
    std::vector<T> mydata(totalNumCoefs);

    if (init_random) {
      auto& locCoefData = spline.single_coef_mirror;
      for(int i = 0; i < nSplines; i++) {
	// note, this could be a different order than in the other code
	myrandom.generate_uniform(&mydata[0], totalNumCoefs);
	int idx = 0;
	for (int ix = 0; ix < locCoefData.extent(0); ix++) {
	  for (int iy = 0; iy < locCoefData.extent(1); iy++) {
	    for (int iz = 0; iz < locCoefData.extent(2); iz++) {
	      locCoefData(ix,iy,iz) = mydata[idx];
	      idx++;
	    }
	  }
	}
	spline.pushCoefToDevice(i);
      }
    }
    //delete[] mydata;
  }

  /** evaluate psi */
  inline void evaluate_v(const PosType& p)
  {
    ScopedTimer local_timer(timer);
    auto u = Lattice.toUnit_floor(p);
    spline.evaluate_v(u[0], u[1], u[2], psi);
  }

  /** evaluate psi, grad and hess */
  inline void evaluate_vgh(const PosType& p)
  {
    ScopedTimer local_timer(timer);
    auto u = Lattice.toUnit_floor(p);
    spline.evaluate_vgh(u[0], u[1], u[2], psi, grad, hess);
  }

  void print(std::ostream& os)
  {
    os << " nSplines=" << nSplines << " nSplinesPerBlock=" << blockSize << std::endl;
  }
};
} // namespace qmcplusplus

#endif
