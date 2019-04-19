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
#include "Utilities/SIMD/allocator.hpp"
#include <Numerics/Containers.h>
#include <limits>
#include <bitset>

namespace qmcplusplus
{
/** enumerator for DistanceTableData::DTType
 *
 * - DT_AOS Use original AoS type
 * - DT_SOA Use SoA type
 * - DT_AOS_PREFERRED Create AoS type, if possible.
 * - DT_SOA_PREFERRED Create SoA type, if possible.
 * The first user of each pair will decide the type of distance table.
 * It is the responsibility of the user class to check DTType.
 */
enum DistTableType
{
  DT_AOS = 0,
  DT_SOA,
  DT_AOS_PREFERRED,
  DT_SOA_PREFERRED
};

/** @ingroup nnlist
 * @brief Abstract class to manage pair data between two ParticleSets.
 *
 * Each DistanceTableData object is fined by Source and Target of ParticleSet
 * types.
 */
struct DistanceTableData
{
  constexpr static unsigned DIM = OHMMS_DIM;

  /**enum for index ordering and storage.
   *@brief Equivalent to using three-dimensional array with (i,j,k)
   * for i = source particle index (slowest),
   *     j = target particle index
   *     k = copies (walkers) index.
   */
  enum
  {
    WalkerIndex = 0,
    SourceIndex,
    VisitorIndex,
    PairIndex
  };
#if (__cplusplus >= 201103L)
  using IndexType       = QMCTraits::IndexType;
  using RealType        = QMCTraits::RealType;
  using PosType         = QMCTraits::PosType;
  using IndexVectorType = aligned_vector<IndexType>;
  using ripair          = std::pair<RealType, IndexType>;
  using RowContainer    = VectorSoAContainer<RealType, DIM>;
#else
  typedef QMCTraits::IndexType IndexType;
  typedef QMCTraits::RealType RealType;
  typedef QMCTraits::PosType PosType;
  typedef aligned_vector<IndexType> IndexVectorType;
  typedef std::pair<RealType, IndexType> ripair;
  typedef Container<RealType, DIM> RowContainer;
#endif

  /// type of cell
  int CellType;
  /// Type of DT
  int DTType;
  /// size of indicies
  TinyVector<IndexType, 4> N;

  /**defgroup SoA data */
  /*@{*/
  /** Distances[i][j] , [Nsources][Ntargets] */
  Matrix<RealType, aligned_allocator<RealType>> Distances;

  /** Displacements[Nsources]x[3][Ntargets] */
  std::vector<RowContainer> Displacements;

  /// actual memory for Displacements
  aligned_vector<RealType> memoryPool;

  /** temp_r */
  aligned_vector<RealType> Temp_r;

  /** temp_dr */
  RowContainer Temp_dr;

  /** true, if full table is needed at loadWalker */
  bool Need_full_table_loadWalker;
  /*@}*/

  /// name of the table
  std::string Name;
  /// constructor using source and target ParticleSet
  DistanceTableData(const ParticleSet& source, const ParticleSet& target)
      : Origin(&source), N(0), Need_full_table_loadWalker(false)
  {}

  /// virutal destructor
  virtual ~DistanceTableData() {}

  /// return the name of table
  inline std::string getName() const { return Name; }
  /// set the name of table
  inline void setName(const std::string& tname) { Name = tname; }

  /// returns the reference the origin particleset
  const ParticleSet& origin() const { return *Origin; }
  inline void reset(const ParticleSet* newcenter) { Origin = newcenter; }

  inline bool is_same_type(int dt_type) const { return DTType == dt_type; }

  /// returns the number of centers
  inline IndexType centers() const { return Origin->getTotalNum(); }

  /// returns the number of centers
  inline IndexType targets() const { return N[VisitorIndex]; }

  /// returns the size of each dimension using enum
  inline IndexType size(int i) const { return N[i]; }

  /// evaluate the Distance Table using only with position array
  virtual void evaluate(ParticleSet& P) = 0;

  /// evaluate the Distance Table
  virtual void evaluate(ParticleSet& P, int jat) = 0;

  /// evaluate the temporary pair relations
  virtual void move(const ParticleSet& P, const PosType& rnew) = 0;

  /// update the distance table by the pair relations
  virtual void update(IndexType jat) = 0;

  const ParticleSet* Origin;
};
} // namespace qmcplusplus
#endif
