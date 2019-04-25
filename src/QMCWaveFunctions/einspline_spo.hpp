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
  
  /** alternative to hand in psi rather than use the integrated one **/
  inline void evaluate_v(const PosType& p, vContainer_type& inPsi) {
    ScopedTimer local_timer(timer);
    auto u = Lattice.toUnit_floor(p);
    spline.evaluate_v(u[0], u[1], u[2], inPsi);
  }    

  template<typename apsdType>
  inline void multi_evaluate_v(Kokkos::View<double**[3]>& all_pos, Kokkos::View<ValueType***> allPsiV, apsdType& apsd) {
    // need to do to_unit_floor for all positions then pass to spline.multi_evaluate_v
    auto& tmpAllPos = all_pos;
    auto& tmpapsd = apsd;
    auto& tmpallPsiV = allPsiV;
    Kokkos::View<double**[3]> allPosToUnitFloor("allPosToUnitFloor", all_pos.extent(0),all_pos.extent(1));
    Kokkos::parallel_for("positionsToFloorLoop", 
			 Kokkos::MDRangePolicy<Kokkos::Rank<2,Kokkos::Iterate::Left> >({0,0}, {tmpAllPos.extent(0), tmpAllPos.extent(1)}),
			 KOKKOS_LAMBDA(const int& walkerNum, const int& knotNum) {
			   apsd(walkerNum).toUnit_floor(tmpAllPos(walkerNum, knotNum, 0),
							tmpAllPos(walkerNum, knotNum, 1),
							tmpAllPos(walkerNum, knotNum, 2),
							allPosToUnitFloor(walkerNum, knotNum, 0),
							allPosToUnitFloor(walkerNum, knotNum, 1),
							allPosToUnitFloor(walkerNum, knotNum, 2));
			     });
    spline.multi_evaluate_v2d(allPosToUnitFloor, tmpallPsiV);
  }			     			     
  
  inline void multi_evaluate_v(std::vector<PosType>& pos_list, std::vector<vContainer_type>& vals) {
    Kokkos::View<vContainer_type*> allPsi("allPsi", vals.size());
    auto allPsiMirror = Kokkos::create_mirror_view(allPsi);
    for (int i = 0; i < vals.size(); i++) {
      allPsiMirror(i) = vals[i];
    }
    Kokkos::deep_copy(allPsi, allPsiMirror);

    // do this in soa spirit
    Kokkos::View<double*[3],Kokkos::LayoutLeft> allPos("allPos", pos_list.size());
    auto allPosMirror = Kokkos::create_mirror_view(allPos);
    for (int i = 0; i < pos_list.size(); i++) {
      auto u = Lattice.toUnit_floor(pos_list[i]);
      allPosMirror(i,0) = u[0];
      allPosMirror(i,1) = u[1];
      allPosMirror(i,2) = u[2];
    }
    Kokkos::deep_copy(allPos, allPosMirror);
    spline.multi_evaluate_v(allPos, allPsi);
  }

  /** evaluate psi, grad and hess */
  inline void evaluate_vgh(const PosType& p)
  {
    ScopedTimer local_timer(timer);
    auto u = Lattice.toUnit_floor(p);
    spline.evaluate_vgh(u[0], u[1], u[2], psi, grad, hess);
  }

  /** alternative to hand in psi rather than use the integrated one **/
  inline void evaluate_vgh(const PosType& p, vContainer_type& inPsi,
			   gContainer_type& inGrad, hContainer_type& inHess)
  {
    ScopedTimer local_timer(timer);
    auto u = Lattice.toUnit_floor(p);
    spline.evaluate_vgh(u[0], u[1], u[2], inPsi, inGrad, inHess);
  }

  // the hope / point here is that everything will be a cheap shallow copy
  // if that is not the case, probably need a rethink

  

  inline void multi_evaluate_vgh(Kokkos::View<double*[3],Kokkos::LayoutLeft>& pos_list, std::vector<vContainer_type>& vals, 
				 std::vector<gContainer_type>& grads, std::vector<hContainer_type>& hesss) {
    Kokkos::View<vContainer_type*> allPsi("allPsi", vals.size());
    Kokkos::View<gContainer_type*> allGrad("allGrad", grads.size());
    Kokkos::View<hContainer_type*> allHess("allHess", hesss.size());
    auto allPsiMirror = Kokkos::create_mirror_view(allPsi);
    auto allGradMirror = Kokkos::create_mirror_view(allGrad);
    auto allHessMirror = Kokkos::create_mirror_view(allHess);
    
    for (int i = 0; i < vals.size(); i++) {
      allPsiMirror(i) = vals[i];
      allGradMirror(i) = grads[i];
      allHessMirror(i) = hesss[i];
    }
    Kokkos::deep_copy(allPsi, allPsiMirror);
    Kokkos::deep_copy(allGrad, allGradMirror);
    Kokkos::deep_copy(allHess, allHessMirror);

    spline.multi_evaluate_vgh(pos_list, allPsi, allGrad, allHess);
  }

  inline void multi_evaluate_vgh(std::vector<PosType>& pos_list, std::vector<vContainer_type>& vals, 
				 std::vector<gContainer_type>& grads, std::vector<hContainer_type>& hesss) {
    // do this in soa spirit
    Kokkos::View<double*[3],Kokkos::LayoutLeft> allPos("allPos", pos_list.size());
    auto allPosMirror = Kokkos::create_mirror_view(allPos);
    for (int i = 0; i < pos_list.size(); i++) {
      auto u = Lattice.toUnit_floor(pos_list[i]);
      allPosMirror(i,0) = u[0];
      allPosMirror(i,1) = u[1];
      allPosMirror(i,2) = u[2];
    }
    Kokkos::deep_copy(allPos, allPosMirror);
    multi_evaluate_vgh(allPos, vals, grads, hesss);
  }





  
  void print(std::ostream& os)
  {
    os << " nSplines=" << nSplines << " nSplinesPerBlock=" << blockSize << std::endl;
  }
};
} // namespace qmcplusplus

#endif
