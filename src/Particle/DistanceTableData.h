////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by:
// Jeremy McMinnis, jmcminis@gmail.com,
//    University of Illinois at Urbana-Champaign
// Jeongnim Kim, jeongnim.kim@gmail.com,
//    University of Illinois at Urbana-Champaign
// Jaron T. Krogel, krogeljt@ornl.gov,
//    Oak Ridge National Laboratory
// Mark A. Berrill, berrillma@ornl.gov,
//    Oak Ridge National Laboratory
//
// File created by:
// Jeongnim Kim, jeongnim.kim@gmail.com,
//    University of Illinois at Urbana-Champaign
////////////////////////////////////////////////////////////////////////////////

#ifndef QMCPLUSPLUS_DISTANCETABLEDATAIMPL_H
#define QMCPLUSPLUS_DISTANCETABLEDATAIMPL_H

#include "Particle/ParticleSet.h"
#include "Utilities/PooledData.h"
#include "Numerics/OhmmsPETE/OhmmsVector.h"
#include "Numerics/OhmmsPETE/OhmmsMatrix.h"
#include <Numerics/Containers.h>
#include <limits>
#include <bitset>

namespace qmcplusplus
{
/** @ingroup nnlist
 * @brief Abstract class to manage pair data between two ParticleSets.
 *
 * Each DistanceTableData object is fined by Source and Target of ParticleSet
 * types.
 */
class DistanceTableData
{
public:
  constexpr static unsigned DIM = OHMMS_DIM;

  using IndexType       = QMCTraits::IndexType;
  using RealType        = QMCTraits::RealType;
  using PosType         = QMCTraits::PosType;
  using IndexVectorType = aligned_vector<IndexType>;
  using RowContainer    = VectorSoAContainer<RealType, DIM>;

  const ParticleSet* Origin;

  /**defgroup SoA data */
  /*@{*/
  /** Distances[i][j] , [Nsources][Ntargets] */
  Matrix<RealType, aligned_allocator<RealType>> Distances;

  /** Displacements[Nsources]x[3][Ntargets] */
  std::vector<RowContainer> Displacements;

  /** temp_r */
  aligned_vector<RealType> Temp_r;

  /** temp_dr */
  RowContainer Temp_dr;
  /*@}*/

protected:
  /// actual memory for Displacements
  aligned_vector<RealType> memoryPool;

  /** true, if full table is needed at loadWalker */
  bool need_full_table_;

  const int Nsources;
  const int Ntargets;

  /// name of the table
  const std::string name_;

public:
  /// constructor using source and target ParticleSet
  DistanceTableData(const ParticleSet& source, const ParticleSet& target)
      : Origin(&source), need_full_table_(false),
        Nsources(source.getTotalNum()),
        Ntargets(target.getTotalNum()),
        name_(source.getName() + "_" + target.getName())
  {}

  /// virutal destructor
  virtual ~DistanceTableData() {}

  /// return the name of table
  inline const std::string& getName() const { return name_; }

  /// returns the reference the origin particleset
  const ParticleSet& origin() const { return *Origin; }
  inline void reset(const ParticleSet* newcenter) { Origin = newcenter; }

  /// returns the number of centers
  inline IndexType centers() const { return Origin->getTotalNum(); }

  /// returns the number of centers
  /// evaluate the Distance Table using only with position array
  virtual void evaluate(ParticleSet& P) = 0;

  /// evaluate the Distance Table
  virtual void evaluate(ParticleSet& P, int jat) = 0;

  /// evaluate the temporary pair relations
  virtual void move(const ParticleSet& P, const PosType& rnew) = 0;

  /// update the distance table by the pair relations
  virtual void update(IndexType jat) = 0;

  ///get need_full_table_
  inline bool getFullTableNeeds() const { return need_full_table_; }

  ///set need_full_table_
  inline void setFullTableNeeds(bool is_needed) { need_full_table_ = is_needed; }
};
} // namespace qmcplusplus
#endif
