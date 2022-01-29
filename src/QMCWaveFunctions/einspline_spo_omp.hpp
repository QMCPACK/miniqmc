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
/** @file einspline_spo_omp.hpp
 */
#ifndef QMCPLUSPLUS_EINSPLINE_SPO_OMP_HPP
#define QMCPLUSPLUS_EINSPLINE_SPO_OMP_HPP
#include <Utilities/Configuration.h>
#include <Utilities/NewTimer.h>
#include <Particle/ParticleSet.h>
#include <Numerics/Spline2/BsplineAllocator.hpp>
#include <Numerics/Containers.h>
#include <PinnedAllocator.h>
#include <OMPTarget/OMPallocator.hpp>
#include "QMCWaveFunctions/SPOSet.h"

namespace qmcplusplus
{
template<typename T>
struct einspline_spo_omp : public SPOSet
{
  template<typename DT>
  using OffloadAllocator = OMPallocator<DT, aligned_allocator<DT>>;
  template<typename DT>
  using OffloadPinnedAllocator = OMPallocator<DT, PinnedAlignedAllocator<DT>>;

  /// define the einsplie data object type
  using self_type   = einspline_spo_omp<T>;
  using spline_type = typename bspline_traits<T, 3>::SplineType;
  using lattice_type            = CrystalLattice<T, 3>;

  /// number of blocks
  int nBlocks;
  /// first logical block index
  int firstBlock;
  /// last gical block index
  int lastBlock;
  /// number of splines
  int nSplines;
  /// number of splines per block
  int nSplinesPerBlock;
  /// if true, responsible for cleaning up einsplines
  bool Owner;
  /// total dimention of v, g, h
  static const int vgh_dim = 10;
  lattice_type Lattice;
  /// use allocator
  BsplineAllocator<T> myAllocator;

  Vector<spline_type*, OffloadAllocator<spline_type*>> einsplines;
  ///thread private ratios for reduction when using nested threading, numVP x numThread
  std::vector<Matrix<T, OffloadPinnedAllocator<T>>> ratios_private;
  ///offload scratch space, dynamically resized to the maximal need
  std::vector<Matrix<T, OffloadPinnedAllocator<T>>> offload_scratch;
  ///offload scratch space for multi_XXX, dynamically resized to the maximal need
  std::vector<Matrix<T, OffloadPinnedAllocator<T>>> multi_offload_scratch;
  ///psiinv and position scratch space, used to avoid allocation on the fly and faster transfer
  Vector<T, OffloadPinnedAllocator<T>> psiinv_pos_copy;

  // for shadows
  Vector<T, OffloadPinnedAllocator<T>> pos_scratch;

  /// Timer
  NewTimer& timer;

  /// default constructor
  einspline_spo_omp();
  /// disable copy constructor
  einspline_spo_omp(const einspline_spo_omp& in) = delete;
  /// disable copy operator
  einspline_spo_omp& operator=(const einspline_spo_omp& in) = delete;

  /** copy constructor
   * @param in einspline_spo_omp
   * @param team_size number of members in a team
   * @param member_id id of this member in a team
   *
   * Create a view of the big object. A simple blocking & padding  method.
   */
  einspline_spo_omp(const einspline_spo_omp& in, int team_size, int member_id);

  /// destructors
  ~einspline_spo_omp();

  /// resize the containers
  void resize();

  // fix for general num_splines
  void set(int nx, int ny, int nz, int num_splines, int nblocks, bool init_random = true);

  /** evaluate psi */
  void evaluate_v(const ParticleSet& P, int iat);

  void evaluate(const ParticleSet& P, int iat, ValueVector_t& psi_v) override;

  void evaluateDetRatios(const VirtualParticleSet& VP,
                         ValueVector_t& psi,
                         const ValueVector_t& psiinv,
                         std::vector<ValueType>& ratios) override;

  /** evaluate psi, grad and lap */
  void evaluate_vgl(const ParticleSet& P, int iat);

  /** evaluate psi, grad and hess */
  void evaluate_vgh(const ParticleSet& P, int iat);

  void evaluate(const ParticleSet& P,
                int iat,
                ValueVector_t& psi_v,
                GradVector_t& dpsi_v,
                ValueVector_t& d2psi_v) override;

  void evaluate_build_vgl(ValueVector_t& psi_v, GradVector_t& dpsi_v, ValueVector_t& d2psi_v);

  /** evaluate psi, grad and hess of multiple walkers with offload */
  void multi_evaluate_vgh(const std::vector<SPOSet*>& spo_list, const std::vector<ParticleSet*>& P_list, int iat);

  void multi_evaluate_ratio_grads(const std::vector<SPOSet*>& spo_list, const std::vector<ParticleSet*>& P_list, int iat);

  void multi_evaluate(const std::vector<SPOSet*>& spo_list,
                      const std::vector<ParticleSet*>& P_list,
                      int iat,
                      const std::vector<ValueVector_t*>& psi_v_list,
                      const std::vector<GradVector_t*>& dpsi_v_list,
                      const std::vector<ValueVector_t*>& d2psi_v_list) override;

  void print(std::ostream& os);
};

extern template struct einspline_spo_omp<OHMMS_PRECISION>;
} // namespace qmcplusplus

#endif
