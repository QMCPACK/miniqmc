// This file is distributed under the University of Illinois/NCSA Open Source
// License. See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//
// File created by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//////////////////////////////////////////////////////////////////////////////////////

#ifndef QMCPLUSPLUS_EEIJASTROW_H
#define QMCPLUSPLUS_EEIJASTROW_H
#include "Utilities/Configuration.h"
#include "QMCWaveFunctions/WaveFunctionComponent.h"
#include "Particle/DistanceTableData.h"
#include <Utilities/SIMD/allocator.hpp>
#include <Utilities/SIMD/algorithm.hpp>
#include <numeric>

namespace qmcplusplus
{
/** @ingroup WaveFunctionComponent
 *  @brief Specialization for three-body Jastrow function using multiple
 *functors
 *
 *Each pair-type can have distinct function \f$u(r_{ij})\f$.
 *For electrons, distinct pair correlation functions are used
 *for spins up-up/down-down and up-down/down-up.
 */
template<class FT>
class ThreeBodyJastrow : public WaveFunctionComponent
{
  /// type of each component U, dU, d2U;
  using valT = typename FT::real_type;
  /// element position type
  using posT = TinyVector<valT, OHMMS_DIM>;
  /// use the same container
  using RowContainer = DistanceTableData::RowContainer;
  /// table index for i-el, el-el is always zero
  int myTableID;
  // nuber of particles
  int Nelec, Nion;
  /// number of particles + padded
  size_t Nelec_padded;
  // number of groups of the target particleset
  int eGroups, iGroups;
  /// reference to the sources (ions)
  const ParticleSet& Ions;
  /// diff value
  RealType DiffVal;

  ///\f$Uat[i] = sum_(j) u_{i,j}\f$
  Vector<valT> Uat, oldUk, newUk;
  ///\f$dUat[i] = sum_(j) du_{i,j}\f$
  using gContainer_type = VectorSoAContainer<valT, OHMMS_DIM>;
  gContainer_type dUat, olddUk, newdUk;
  ///\f$d2Uat[i] = sum_(j) d2u_{i,j}\f$
  Vector<valT> d2Uat, oldd2Uk, newd2Uk;
  /// current values during PbyP
  valT cur_Uat, cur_d2Uat;
  posT cur_dUat, dUat_temp;
  /// container for the Jastrow functions
  Array<FT*, 3> F;

  /// the cutoff for e-I pairs
  std::vector<valT> Ion_cutoff;
  /// the electrons around ions within the cutoff radius, grouped by species
  Array<std::vector<int>, 2> elecs_inside;
  Array<std::vector<valT>, 2> elecs_inside_dist;
  Array<std::vector<posT>, 2> elecs_inside_displ;
  /// the ions around
  std::vector<int> ions_nearby;

  /// work buffer size
  size_t Nbuffer;
  /// compressed distances
  aligned_vector<valT> Distjk_Compressed, DistkI_Compressed, DistjI_Compressed;
  std::vector<int> DistIndice_k;
  /// compressed displacements
  gContainer_type Disp_jk_Compressed, Disp_jI_Compressed, Disp_kI_Compressed;
  /// work result buffer
  VectorSoAContainer<valT, 9> mVGL;

public:
  /// alias FuncType
  using FuncType = FT;

  ThreeBodyJastrow(const ParticleSet& ions, ParticleSet& elecs, bool is_master = false) : Ions(ions)
  {
    WaveFunctionComponentName                               = "ThreeBodyJastrow";
    myTableID                                               = elecs.addTable(Ions, DT_SOA);
    elecs.DistTables[myTableID]->Need_full_table_loadWalker = true;
    init(elecs);
  }

  ~ThreeBodyJastrow() {}

  void init(ParticleSet& p)
  {
    Nelec        = p.getTotalNum();
    Nelec_padded = getAlignedSize<valT>(Nelec);
    Nion         = Ions.getTotalNum();
    iGroups      = Ions.getSpeciesSet().getTotalNum();
    eGroups      = p.groups();

    Uat.resize(Nelec);
    dUat.resize(Nelec);
    d2Uat.resize(Nelec);

    oldUk.resize(Nelec);
    olddUk.resize(Nelec);
    oldd2Uk.resize(Nelec);
    newUk.resize(Nelec);
    newdUk.resize(Nelec);
    newd2Uk.resize(Nelec);

    F.resize(iGroups, eGroups, eGroups);
    F = nullptr;
    elecs_inside.resize(eGroups, Nion);
    elecs_inside_dist.resize(eGroups, Nion);
    elecs_inside_displ.resize(eGroups, Nion);
    ions_nearby.resize(Nion);
    Ion_cutoff.resize(Nion, 0.0);

    // initialize buffers
    Nbuffer = Nelec;
    mVGL.resize(Nbuffer);
    Distjk_Compressed.resize(Nbuffer);
    DistjI_Compressed.resize(Nbuffer);
    DistkI_Compressed.resize(Nbuffer);
    Disp_jk_Compressed.resize(Nbuffer);
    Disp_jI_Compressed.resize(Nbuffer);
    Disp_kI_Compressed.resize(Nbuffer);
    DistIndice_k.resize(Nbuffer);
  }

  void addFunc(int iSpecies, int eSpecies1, int eSpecies2, FT* j)
  {
    if (eSpecies1 == eSpecies2)
    {
      // if only up-up is specified, assume spin-unpolarized correlations
      if (eSpecies1 == 0)
        for (int eG1 = 0; eG1 < eGroups; eG1++)
          for (int eG2 = 0; eG2 < eGroups; eG2++)
          {
            if (F(iSpecies, eG1, eG2) == 0)
              F(iSpecies, eG1, eG2) = j;
          }
    }
    else
    {
      F(iSpecies, eSpecies1, eSpecies2) = j;
      F(iSpecies, eSpecies2, eSpecies1) = j;
    }
    if (j)
    {
      RealType rcut = 0.5 * j->cutoff_radius;
      for (int i = 0; i < Nion; i++)
        if (Ions.GroupID[i] == iSpecies)
          Ion_cutoff[i] = rcut;
    }
    else
    {
      APP_ABORT("ThreeBodyJastrow::addFunc  Jastrow function pointer is NULL");
    }
  }

  /** check that correlation information is complete
   */
  void check_complete()
  {
    // check that correlation pointers are either all 0 or all assigned
    bool complete = true;
    for (int i = 0; i < iGroups; ++i)
    {
      int nfilled = 0;
      bool partial;
      for (int e1 = 0; e1 < eGroups; ++e1)
        for (int e2 = 0; e2 < eGroups; ++e2)
          if (F(i, e1, e2) != 0)
            nfilled++;
      partial = nfilled > 0 && nfilled < eGroups * eGroups;
      if (partial)
        app_log() << "J3 eeI is missing correlation for ion " << i << std::endl;
      complete = complete && !partial;
    }
    if (!complete)
    {
      APP_ABORT("ThreeBodyJastrow::check_complete  J3 eeI is missing "
                "correlation components\n  see preceding messages for details");
    }
    // first set radii
    for (int i = 0; i < Nion; ++i)
    {
      FT* f = F(Ions.GroupID[i], 0, 0);
      if (f != 0)
        Ion_cutoff[i] = .5 * f->cutoff_radius;
    }
    // then check radii
    bool all_radii_match = true;
    for (int i = 0; i < iGroups; ++i)
    {
      if (F(i, 0, 0) != 0)
      {
        bool radii_match = true;
        RealType rcut    = F(i, 0, 0)->cutoff_radius;
        for (int e1 = 0; e1 < eGroups; ++e1)
          for (int e2 = 0; e2 < eGroups; ++e2)
            radii_match = radii_match && F(i, e1, e2)->cutoff_radius == rcut;
        if (!radii_match)
          app_log() << "eeI functors for ion species " << i << " have different radii" << std::endl;
        all_radii_match = all_radii_match && radii_match;
      }
    }
    if (!all_radii_match)
    {
      APP_ABORT("ThreeBodyJastrow::check_radii  J3 eeI are inconsistent for "
                "some ion species\n  see preceding messages for details");
    }
  }

  void build_compact_list(ParticleSet& P)
  {
    const DistanceTableData& eI_table = (*P.DistTables[myTableID]);

    for (int iat = 0; iat < Nion; ++iat)
      for (int jg = 0; jg < eGroups; ++jg)
      {
        elecs_inside(jg, iat).clear();
        elecs_inside_dist(jg, iat).clear();
        elecs_inside_displ(jg, iat).clear();
      }

    for (int jg = 0; jg < eGroups; ++jg)
      for (int jel = P.first(jg); jel < P.last(jg); jel++)
        for (int iat = 0; iat < Nion; ++iat)
          if (eI_table.Distances[jel][iat] < Ion_cutoff[iat])
          {
            elecs_inside(jg, iat).push_back(jel);
            elecs_inside_dist(jg, iat).push_back(eI_table.Distances[jel][iat]);
            elecs_inside_displ(jg, iat).push_back(eI_table.Displacements[jel][iat]);
          }
  }

  RealType evaluateLog(ParticleSet& P,
                       ParticleSet::ParticleGradient_t& G,
                       ParticleSet::ParticleLaplacian_t& L)
  {
    evaluateGL(P, G, L, true);
    return LogValue;
  }

  ValueType ratio(ParticleSet& P, int iat)
  {
    UpdateMode = ORB_PBYP_RATIO;

    const DistanceTableData& eI_table = (*P.DistTables[myTableID]);
    const DistanceTableData& ee_table = (*P.DistTables[0]);
    cur_Uat = computeU(P, iat, P.GroupID[iat], eI_table.Temp_r.data(), ee_table.Temp_r.data());
    DiffVal = Uat[iat] - cur_Uat;
    return std::exp(DiffVal);
  }

  GradType evalGrad(ParticleSet& P, int iat) { return GradType(dUat[iat]); }

  ValueType ratioGrad(ParticleSet& P, int iat, GradType& grad_iat)
  {
    UpdateMode = ORB_PBYP_PARTIAL;

    const DistanceTableData& eI_table = (*P.DistTables[myTableID]);
    const DistanceTableData& ee_table = (*P.DistTables[0]);
    computeU3(P,
              iat,
              eI_table.Temp_r.data(),
              eI_table.Temp_dr,
              ee_table.Temp_r.data(),
              ee_table.Temp_dr,
              cur_Uat,
              cur_dUat,
              cur_d2Uat,
              newUk,
              newdUk,
              newd2Uk);
    DiffVal = Uat[iat] - cur_Uat;
    grad_iat += cur_dUat;
    return std::exp(DiffVal);
  }

  void acceptMove(ParticleSet& P, int iat)
  {
    const DistanceTableData& eI_table = (*P.DistTables[myTableID]);
    const DistanceTableData& ee_table = (*P.DistTables[0]);
    // get the old value, grad, lapl
    computeU3(P,
              iat,
              eI_table.Distances[iat],
              eI_table.Displacements[iat],
              ee_table.Distances[iat],
              ee_table.Displacements[iat],
              Uat[iat],
              dUat_temp,
              d2Uat[iat],
              oldUk,
              olddUk,
              oldd2Uk);
    if (UpdateMode == ORB_PBYP_RATIO)
    { // ratio-only during the move; need to compute derivatives
      computeU3(P,
                iat,
                eI_table.Temp_r.data(),
                eI_table.Temp_dr,
                ee_table.Temp_r.data(),
                ee_table.Temp_dr,
                cur_Uat,
                cur_dUat,
                cur_d2Uat,
                newUk,
                newdUk,
                newd2Uk);
    }

    for (int jel = 0; jel < Nelec; jel++)
    {
      Uat[jel] += newUk[jel] - oldUk[jel];
      d2Uat[jel] += newd2Uk[jel] - oldd2Uk[jel];
    }
    for (int idim = 0; idim < OHMMS_DIM; ++idim)
    {
      valT* restrict save_g      = dUat.data(idim);
      const valT* restrict new_g = newdUk.data(idim);
      const valT* restrict old_g = olddUk.data(idim);
      for (int jel = 0; jel < Nelec; jel++)
        save_g[jel] += new_g[jel] - old_g[jel];
    }

    LogValue += Uat[iat] - cur_Uat;
    Uat[iat]   = cur_Uat;
    dUat(iat)  = cur_dUat;
    d2Uat[iat] = cur_d2Uat;

    const int ig = P.GroupID[iat];
    // update compact list elecs_inside
    for (int jat = 0; jat < Nion; jat++)
    {
      bool inside = eI_table.Temp_r[jat] < Ion_cutoff[jat];
      auto iter   = find(elecs_inside(ig, jat).begin(), elecs_inside(ig, jat).end(), iat);
      auto iter_dist =
          elecs_inside_dist(ig, jat).begin() + std::distance(elecs_inside(ig, jat).begin(), iter);
      auto iter_displ =
          elecs_inside_displ(ig, jat).begin() + std::distance(elecs_inside(ig, jat).begin(), iter);
      if (inside)
      {
        if (iter == elecs_inside(ig, jat).end())
        {
          elecs_inside(ig, jat).push_back(iat);
          elecs_inside_dist(ig, jat).push_back(eI_table.Temp_r[jat]);
          elecs_inside_displ(ig, jat).push_back(eI_table.Temp_dr[jat]);
        }
        else
        {
          *iter_dist  = eI_table.Temp_r[jat];
          *iter_displ = eI_table.Temp_dr[jat];
        }
      }
      else
      {
        if (iter != elecs_inside(ig, jat).end())
        {
          *iter = elecs_inside(ig, jat).back();
          elecs_inside(ig, jat).pop_back();
          *iter_dist = elecs_inside_dist(ig, jat).back();
          elecs_inside_dist(ig, jat).pop_back();
          *iter_displ = elecs_inside_displ(ig, jat).back();
          elecs_inside_displ(ig, jat).pop_back();
        }
      }
    }
  }

  inline void recompute(ParticleSet& P)
  {
    const DistanceTableData& eI_table = (*P.DistTables[myTableID]);
    const DistanceTableData& ee_table = (*P.DistTables[0]);

    build_compact_list(P);

    for (int jel = 0; jel < Nelec; ++jel)
    {
      computeU3(P,
                jel,
                eI_table.Distances[jel],
                eI_table.Displacements[jel],
                ee_table.Distances[jel],
                ee_table.Displacements[jel],
                Uat[jel],
                dUat_temp,
                d2Uat[jel],
                newUk,
                newdUk,
                newd2Uk,
                true);
      dUat(jel) = dUat_temp;
      // add the contribution from the upper triangle
      for (int kel = 0; kel < jel; kel++)
      {
        Uat[kel] += newUk[kel];
        d2Uat[kel] += newd2Uk[kel];
      }
      for (int idim = 0; idim < OHMMS_DIM; ++idim)
      {
        valT* restrict save_g      = dUat.data(idim);
        const valT* restrict new_g = newdUk.data(idim);
        for (int kel = 0; kel < jel; kel++)
          save_g[kel] += new_g[kel];
      }
    }
  }

  inline valT
      computeU(const ParticleSet& P, int jel, int jg, const RealType* distjI, const RealType* distjk)
  {
    const DistanceTableData& eI_table = (*P.DistTables[myTableID]);

    ions_nearby.clear();
    for (int iat = 0; iat < Nion; ++iat)
      if (distjI[iat] < Ion_cutoff[iat])
        ions_nearby.push_back(iat);

    valT Uj = valT(0);
    for (int kg = 0; kg < eGroups; ++kg)
    {
      int kel_counter = 0;
      for (int iind = 0; iind < ions_nearby.size(); ++iind)
      {
        const int iat   = ions_nearby[iind];
        const int ig    = Ions.GroupID[iat];
        const valT r_jI = distjI[iat];
        for (int kind = 0; kind < elecs_inside(kg, iat).size(); kind++)
        {
          const int kel = elecs_inside(kg, iat)[kind];
          if (kel != jel)
          {
            DistkI_Compressed[kel_counter] = elecs_inside_dist(kg, iat)[kind];
            Distjk_Compressed[kel_counter] = distjk[kel];
            DistjI_Compressed[kel_counter] = r_jI;
            kel_counter++;
            if (kel_counter == Nbuffer)
            {
              const FT& feeI(*F(ig, jg, kg));
              Uj += feeI.evaluateV(kel_counter,
                                   Distjk_Compressed.data(),
                                   DistjI_Compressed.data(),
                                   DistkI_Compressed.data());
              kel_counter = 0;
            }
          }
        }
        if ((iind + 1 == ions_nearby.size() || ig != Ions.GroupID[ions_nearby[iind + 1]]) &&
            kel_counter > 0)
        {
          const FT& feeI(*F(ig, jg, kg));
          Uj += feeI.evaluateV(kel_counter,
                               Distjk_Compressed.data(),
                               DistjI_Compressed.data(),
                               DistkI_Compressed.data());
          kel_counter = 0;
        }
      }
    }
    return Uj;
  }

  inline void computeU3_engine(const ParticleSet& P,
                               const FT& feeI,
                               int kel_counter,
                               valT& Uj,
                               posT& dUj,
                               valT& d2Uj,
                               Vector<valT>& Uk,
                               gContainer_type& dUk,
                               Vector<valT>& d2Uk)
  {
    const DistanceTableData& eI_table = (*P.DistTables[myTableID]);

    constexpr valT czero(0);
    constexpr valT cone(1);
    constexpr valT ctwo(2);
    constexpr valT lapfac = OHMMS_DIM - cone;

    valT* restrict val     = mVGL.data(0);
    valT* restrict gradF0  = mVGL.data(1);
    valT* restrict gradF1  = mVGL.data(2);
    valT* restrict gradF2  = mVGL.data(3);
    valT* restrict hessF00 = mVGL.data(4);
    valT* restrict hessF11 = mVGL.data(5);
    valT* restrict hessF22 = mVGL.data(6);
    valT* restrict hessF01 = mVGL.data(7);
    valT* restrict hessF02 = mVGL.data(8);

    feeI.evaluateVGL(kel_counter,
                     Distjk_Compressed.data(),
                     DistjI_Compressed.data(),
                     DistkI_Compressed.data(),
                     val,
                     gradF0,
                     gradF1,
                     gradF2,
                     hessF00,
                     hessF11,
                     hessF22,
                     hessF01,
                     hessF02);

    // compute the contribution to jel, kel
    Uj               = simd::accumulate_n(val, kel_counter, Uj);
    valT gradF0_sum  = simd::accumulate_n(gradF0, kel_counter, czero);
    valT gradF1_sum  = simd::accumulate_n(gradF1, kel_counter, czero);
    valT hessF00_sum = simd::accumulate_n(hessF00, kel_counter, czero);
    valT hessF11_sum = simd::accumulate_n(hessF11, kel_counter, czero);
    d2Uj -= hessF00_sum + hessF11_sum + lapfac * (gradF0_sum + gradF1_sum);
    std::fill_n(hessF11, kel_counter, czero);
    for (int idim = 0; idim < OHMMS_DIM; ++idim)
    {
      valT* restrict jk = Disp_jk_Compressed.data(idim);
      valT* restrict jI = Disp_jI_Compressed.data(idim);
      valT* restrict kI = Disp_kI_Compressed.data(idim);
      valT dUj_x(0);
      for (int kel_index = 0; kel_index < kel_counter; kel_index++)
      {
        // recycle hessF11
        hessF11[kel_index] += kI[kel_index] * jk[kel_index];
        dUj_x += gradF1[kel_index] * jI[kel_index];
        // destroy jk, kI
        const valT temp = jk[kel_index] * gradF0[kel_index];
        dUj_x += temp;
        jk[kel_index] *= jI[kel_index];
        kI[kel_index] = kI[kel_index] * gradF2[kel_index] - temp;
      }
      dUj[idim] += dUj_x;

      valT* restrict jk0 = Disp_jk_Compressed.data(0);
      if (idim > 0)
      {
        for (int kel_index = 0; kel_index < kel_counter; kel_index++)
          jk0[kel_index] += jk[kel_index];
      }

      valT* restrict dUk_x = dUk.data(idim);
      for (int kel_index = 0; kel_index < kel_counter; kel_index++)
        dUk_x[DistIndice_k[kel_index]] += kI[kel_index];
    }
    valT sum(0);
    valT* restrict jk0 = Disp_jk_Compressed.data(0);
    for (int kel_index = 0; kel_index < kel_counter; kel_index++)
      sum += hessF01[kel_index] * jk0[kel_index];
    d2Uj -= ctwo * sum;

    for (int kel_index = 0; kel_index < kel_counter; kel_index++)
      hessF00[kel_index] = hessF00[kel_index] + hessF22[kel_index] +
          lapfac * (gradF0[kel_index] + gradF2[kel_index]) -
          ctwo * hessF02[kel_index] * hessF11[kel_index];

    for (int kel_index = 0; kel_index < kel_counter; kel_index++)
    {
      const int kel = DistIndice_k[kel_index];
      Uk[kel] += val[kel_index];
      d2Uk[kel] -= hessF00[kel_index];
    }
  }

  inline void computeU3(const ParticleSet& P,
                        int jel,
                        const RealType* distjI,
                        const RowContainer& displjI,
                        const RealType* distjk,
                        const RowContainer& displjk,
                        valT& Uj,
                        posT& dUj,
                        valT& d2Uj,
                        Vector<valT>& Uk,
                        gContainer_type& dUk,
                        Vector<valT>& d2Uk,
                        bool triangle = false)
  {
    constexpr valT czero(0);

    Uj   = czero;
    dUj  = posT();
    d2Uj = czero;

    const int jg = P.GroupID[jel];

    const int kelmax = triangle ? jel : Nelec;
    std::fill_n(Uk.data(), kelmax, czero);
    std::fill_n(d2Uk.data(), kelmax, czero);
    for (int idim = 0; idim < OHMMS_DIM; ++idim)
      std::fill_n(dUk.data(idim), kelmax, czero);

    ions_nearby.clear();
    for (int iat = 0; iat < Nion; ++iat)
      if (distjI[iat] < Ion_cutoff[iat])
        ions_nearby.push_back(iat);

    for (int kg = 0; kg < eGroups; ++kg)
    {
      int kel_counter = 0;
      for (int iind = 0; iind < ions_nearby.size(); ++iind)
      {
        const int iat      = ions_nearby[iind];
        const int ig       = Ions.GroupID[iat];
        const valT r_jI    = distjI[iat];
        const posT disp_Ij = displjI[iat];
        for (int kind = 0; kind < elecs_inside(kg, iat).size(); kind++)
        {
          const int kel = elecs_inside(kg, iat)[kind];
          if (kel < kelmax && kel != jel)
          {
            DistkI_Compressed[kel_counter]  = elecs_inside_dist(kg, iat)[kind];
            DistjI_Compressed[kel_counter]  = r_jI;
            Distjk_Compressed[kel_counter]  = distjk[kel];
            Disp_kI_Compressed(kel_counter) = elecs_inside_displ(kg, iat)[kind];
            Disp_jI_Compressed(kel_counter) = disp_Ij;
            Disp_jk_Compressed(kel_counter) = displjk[kel];
            DistIndice_k[kel_counter]       = kel;
            kel_counter++;
            if (kel_counter == Nbuffer)
            {
              const FT& feeI(*F(ig, jg, kg));
              computeU3_engine(P, feeI, kel_counter, Uj, dUj, d2Uj, Uk, dUk, d2Uk);
              kel_counter = 0;
            }
          }
        }
        if ((iind + 1 == ions_nearby.size() || ig != Ions.GroupID[ions_nearby[iind + 1]]) &&
            kel_counter > 0)
        {
          const FT& feeI(*F(ig, jg, kg));
          computeU3_engine(P, feeI, kel_counter, Uj, dUj, d2Uj, Uk, dUk, d2Uk);
          kel_counter = 0;
        }
      }
    }
  }

  void evaluateGL(ParticleSet& P,
                  ParticleSet::ParticleGradient_t& G,
                  ParticleSet::ParticleLaplacian_t& L,
                  bool fromscratch = false)
  {
    if (fromscratch)
      recompute(P);
    LogValue = valT(0);
    for (int iat = 0; iat < Nelec; ++iat)
    {
      LogValue += Uat[iat];
      G[iat] += dUat[iat];
      L[iat] += d2Uat[iat];
    }

    constexpr valT mhalf(-0.5);
    LogValue = mhalf * LogValue;
  }
};

} // namespace qmcplusplus
#endif
