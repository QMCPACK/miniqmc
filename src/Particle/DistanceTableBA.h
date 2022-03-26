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
////////////////////////////////////////////////////////////////////////////////
// -*- C++ -*-
#ifndef QMCPLUSPLUS_DTDIMPL_BA_H
#define QMCPLUSPLUS_DTDIMPL_BA_H

namespace qmcplusplus
{
/**@ingroup nnlist
 * @brief A derived classe from DistacneTableData, specialized for AB using a
 * transposed form
 */
template<typename T, unsigned D, int SC>
struct DistanceTableBA : public DTD_BConds<T, D, SC>, public DistanceTableData
{
  int BlockSize;

  DistanceTableBA(const ParticleSet& source, ParticleSet& target)
      : DTD_BConds<T, D, SC>(source.Lattice), DistanceTableData(source, target)
  {
    resize(source.getTotalNum(), target.getTotalNum());
    PRAGMA_OFFLOAD("omp target enter data map(to : this[:1])")
  }

  void resize(int ns, int nt)
  {
    if (Nsources * Ntargets == 0)
      return;

    int Ntargets_padded = getAlignedSize<T>(Ntargets);
    int Nsources_padded = getAlignedSize<T>(Nsources);

    Distances.resize(Ntargets, Nsources_padded);

    BlockSize = Nsources_padded * D;
    memoryPool.resize(Ntargets * BlockSize);
    Displacements.resize(Ntargets);
    for (int i = 0; i < Ntargets; ++i)
      Displacements[i].attachReference(Nsources, Nsources_padded, memoryPool.data() + i * BlockSize);

    Temp_r.resize(Nsources);
    Temp_dr.resize(Nsources);
  }

  DistanceTableBA()                       = delete;
  DistanceTableBA(const DistanceTableBA&) = delete;

  ~DistanceTableBA() { PRAGMA_OFFLOAD("omp target exit data map(delete : this[:1])") }

  /** evaluate the full table */
  inline void evaluate(ParticleSet& P)
  {
    /*
    // be aware of the sign of Displacement
    for (int iat = 0; iat < Ntargets; ++iat)
      DTD_BConds<T, D, SC>::computeDistances(P.R[iat], Origin->RSoA, Distances[iat], Displacements[iat], 0, Nsources);
    */

    const size_t ntgt_local  = Ntargets;
    const size_t ntgt_padded = getAlignedSize<T>(Ntargets);
    const size_t nsrc_padded = getAlignedSize<T>(Nsources);

    auto* dist_ptr  = Distances.data();
    auto* displ_ptr = memoryPool.data();
    auto* src_ptr   = Origin->RSoA.data();
    auto* tgt_ptr   = P.RSoA.data();

    const int ChunkSizePerTeam = 128;
    const size_t num_teams     = (nsrc_padded + ChunkSizePerTeam - 1) / ChunkSizePerTeam;

    PRAGMA_OFFLOAD("omp target teams distribute collapse(2) num_teams(Ntargets * num_teams) \
                    map(to: src_ptr[:D*Origin->RSoA.capacity()], tgt_ptr[:D*P.RSoA.capacity()]) \
                    map(from: dist_ptr[:Distances.size()], displ_ptr[:memoryPool.size()])")
    for (size_t iat = 0; iat < ntgt_local; ++iat)
      for (size_t team_id = 0; team_id < num_teams; ++team_id)
      {
        T pos[D];
        for (int idim = 0; idim < D; idim++)
          pos[idim] = *(tgt_ptr + ntgt_padded * idim + iat);

        const size_t first = ChunkSizePerTeam * team_id;
        const size_t last  = std::min(first + ChunkSizePerTeam, nsrc_padded);

        PRAGMA_OFFLOAD("omp parallel for")
        for (size_t jel = first; jel < last; ++jel)
        {
          DTD_BConds<T, D, SC>::computeDistancesOffload(pos, src_ptr, nsrc_padded, dist_ptr + nsrc_padded * iat,
                                                        displ_ptr + nsrc_padded * D * iat, nsrc_padded, jel);
        }
      }
  }

  /** evaluate the iat-row with the current position
   *
   * Fill Temp_r and Temp_dr and copy them Distances & Displacements
   */
  inline void evaluate(ParticleSet& P, IndexType iat)
  {
    DTD_BConds<T, D, SC>::computeDistances(P.R[iat], Origin->RSoA, Distances[iat], Displacements[iat], 0, Nsources);
  }

  /// evaluate the temporary pair relations
  inline void move(const ParticleSet& P, const PosType& rnew, IndexType iat)
  {
    DTD_BConds<T, D, SC>::computeDistances(rnew, Origin->RSoA, Temp_r.data(), Temp_dr, 0, Nsources);
  }

  /// update the stripe for jat-th particle
  inline void update(IndexType iat)
  {
    std::copy_n(Temp_r.data(), Nsources, Distances[iat]);
    for (int idim = 0; idim < D; ++idim)
      std::copy_n(Temp_dr.data(idim), Nsources, Displacements[iat].data(idim));
  }
};
} // namespace qmcplusplus
#endif
