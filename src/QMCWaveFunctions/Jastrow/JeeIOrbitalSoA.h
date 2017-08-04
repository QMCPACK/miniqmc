////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by:
// Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//
// File created by:
// Ye Luo, yeluo@anl.gov, Argonne National Laboratory
////////////////////////////////////////////////////////////////////////////////

#ifndef QMCPLUSPLUS_EEIJASTROW_OPTIMIZED_SOA_H
#define QMCPLUSPLUS_EEIJASTROW_OPTIMIZED_SOA_H
#include "Configuration.h"
#include "QMCWaveFunctions/OrbitalBase.h"
#include "Particle/DistanceTableData.h"
#include "OhmmsPETE/OhmmsArray.h"
#include <map>
#include <numeric>

/*!
 * @file JeeIOrbitalSoA.h
 */

namespace qmcplusplus
{

/** @ingroup OrbitalComponent
 *  @brief Specialization for three-body Jastrow function using multiple
 *functors
 *
 *Each pair-type can have distinct function \f$u(r_{ij})\f$.
 *For electrons, distinct pair correlation functions are used
 *for spins up-up/down-down and up-down/down-up.
 */
template <class FT> class JeeIOrbitalSoA : public OrbitalBase
{
  /// type of each component U, dU, d2U;
  using valT = typename FT::real_type;
  /// element position type
  using posT = TinyVector<valT, OHMMS_DIM>;
  /// use the same container
  using RowContainer = DistanceTableData::RowContainer;
  /// define container types
  using vContainer_type = aligned_vector<valT>;
  using gContainer_type = VectorSoaContainer<valT, OHMMS_DIM>;
  /// table index for i-el, el-el is always zero
  int myTableID;
  // nuber of particles
  int Nelec, Nion;
  // number of groups of the target particleset
  int eGroups, iGroups;
  /// reference to the sources (ions)
  const ParticleSet &Ions;
  /// diff value
  RealType DiffVal;

  ///\f$Uat[i] = sum_(j) u_{i,j}\f$
  Vector<valT> Uat;
  vContainer_type oldUk, newUk;
  ///\f$dUat[i] = sum_(j) du_{i,j}\f$
  Vector<posT> dUat;
  valT *FirstAddressOfdU, *LastAddressOfdU;
  gContainer_type olddUk, newdUk;
  ///\f$d2Uat[i] = sum_(j) d2u_{i,j}\f$
  Vector<valT> d2Uat;
  vContainer_type oldd2Uk, newd2Uk;
  valT cur_Uat, cur_d2Uat;
  posT cur_dUat;
  /// container for the Jastrow functions
  Array<FT *, 3> F;

  std::map<std::string, FT *> J3Unique;
  // YYYY
  std::map<FT *, int> J3UniqueIndex;

  /// the cutoff for e-I pairs
  std::vector<valT> Ion_cutoff;
  /// the electrons around ions within the cutoff radius, grouped by species
  Array<std::vector<int>, 2> elecs_inside;

  /// compressed distances
  aligned_vector<valT> Distjk_Compressed, DistkI_Compressed;
  std::vector<int> DistIndice;

  using VGL_type = VectorSoaContainer<valT, 9>;
  VGL_type mVGL;

  // Used for evaluating derivatives with respect to the parameters
  int NumVars;

public:
  /// alias FuncType
  using FuncType = FT;

  JeeIOrbitalSoA(const ParticleSet &ions, ParticleSet &elecs,
                 bool is_master = false)
      : Ions(ions), NumVars(0)
  {
    OrbitalName = "JeeIOrbitalSoA";
    myTableID   = elecs.addTable(Ions, DT_SOA);
    elecs.DistTables[myTableID]->Need_full_table_loadWalker = true;
    init(elecs);
  }

  ~JeeIOrbitalSoA() {}

  void init(ParticleSet &p)
  {
    Nelec   = p.getTotalNum();
    Nion    = Ions.getTotalNum();
    iGroups = Ions.getSpeciesSet().getTotalNum();
    eGroups = p.groups();

    Uat.resize(Nelec);
    dUat.resize(Nelec);
    FirstAddressOfdU = &(dUat[0][0]);
    LastAddressOfdU  = FirstAddressOfdU + dUat.size() * OHMMS_DIM;
    d2Uat.resize(Nelec);

    oldUk.resize(Nelec);
    olddUk.resize(Nelec);
    oldd2Uk.resize(Nelec);
    newUk.resize(Nelec);
    newdUk.resize(Nelec);
    newd2Uk.resize(Nelec);

    F.resize(iGroups, eGroups, eGroups);
    F = nullptr;
    elecs_inside.resize(Nion, eGroups);
    Ion_cutoff.resize(Nion);

    mVGL.resize(Nelec);
    DistkI_Compressed.resize(Nelec);
    Distjk_Compressed.resize(Nelec);
    DistIndice.resize(Nelec);
  }

  void addFunc(int iSpecies, int eSpecies1, int eSpecies2, FT *j)
  {
    if (eSpecies1 == eSpecies2)
    {
      // if only up-up is specified, assume spin-unpolarized correlations
      if (eSpecies1 == 0)
        for (int eG1 = 0; eG1 < eGroups; eG1++)
          for (int eG2 = 0; eG2 < eGroups; eG2++)
          {
            if (F(iSpecies, eG1, eG2) == 0) F(iSpecies, eG1, eG2) = j;
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
      for (int i                                       = 0; i < Nion; i++)
        if (Ions.GroupID[i] == iSpecies) Ion_cutoff[i] = rcut;
    }
    else
    {
      APP_ABORT("JeeIOrbitalSoA::addFunc  Jastrow function pointer is NULL");
    }
    std::strstream aname;
    aname << iSpecies << "_" << eSpecies1 << "_" << eSpecies2;
    J3Unique[aname.str()] = j;
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
          if (F(i, e1, e2) != 0) nfilled++;
      partial = nfilled > 0 && nfilled < eGroups * eGroups;
      if (partial)
        app_log() << "J3 eeI is missing correlation for ion " << i << std::endl;
      complete = complete && !partial;
    }
    if (!complete)
    {
      APP_ABORT("JeeIOrbitalSoA::check_complete  J3 eeI is missing correlation "
                "components\n  see preceding messages for details");
    }
    // first set radii
    for (int i = 0; i < Nion; ++i)
    {
      FT *f                     = F(Ions.GroupID[i], 0, 0);
      if (f != 0) Ion_cutoff[i] = .5 * f->cutoff_radius;
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
          app_log() << "eeI functors for ion species " << i
                    << " have different radii" << std::endl;
        all_radii_match = all_radii_match && radii_match;
      }
    }
    if (!all_radii_match)
    {
      APP_ABORT("JeeIOrbitalSoA::check_radii  J3 eeI are inconsistent for some "
                "ion species\n  see preceding messages for details");
    }
  }

  void build_compact_list(ParticleSet &P)
  {
    const DistanceTableData &eI_table = (*P.DistTables[myTableID]);

    for (int iat = 0; iat < Nion; ++iat)
      for (int jg = 0; jg < eGroups; ++jg) elecs_inside(iat, jg).clear();

    for (int jg = 0; jg < eGroups; ++jg)
      for (int jel = P.first(jg); jel < P.last(jg); jel++)
        for (int iat = 0; iat < Nion; ++iat)
          if (eI_table.Distances[jel][iat] < Ion_cutoff[iat])
            elecs_inside(iat, jg).push_back(jel);
  }

  RealType evaluateLog(ParticleSet &P, ParticleSet::ParticleGradient_t &G,
                       ParticleSet::ParticleLaplacian_t &L)
  {
    evaluateGL(P, G, L, true);
    return LogValue;
  }

  ValueType ratio(ParticleSet &P, int iat)
  {
    UpdateMode = ORB_PBYP_RATIO;

    const DistanceTableData &eI_table = (*P.DistTables[myTableID]);
    const DistanceTableData &ee_table = (*P.DistTables[0]);
    cur_Uat = computeU(P, iat, eI_table.Temp_r.data(), ee_table.Temp_r.data());
    DiffVal = Uat[iat] - cur_Uat;
    return std::exp(DiffVal);
  }

  GradType evalGrad(ParticleSet &P, int iat) { return GradType(dUat[iat]); }

  ValueType ratioGrad(ParticleSet &P, int iat, GradType &grad_iat)
  {
    UpdateMode = ORB_PBYP_PARTIAL;

    const DistanceTableData &eI_table = (*P.DistTables[myTableID]);
    const DistanceTableData &ee_table = (*P.DistTables[0]);
    computeU3(P, iat, eI_table.Temp_r.data(), eI_table.Temp_dr,
              ee_table.Temp_r.data(), ee_table.Temp_dr, cur_Uat, cur_dUat,
              cur_d2Uat, newUk, newdUk, newd2Uk);
    DiffVal = Uat[iat] - cur_Uat;
    grad_iat += cur_dUat;
    return std::exp(DiffVal);
  }

  void acceptMove(ParticleSet &P, int iat)
  {
    const DistanceTableData &eI_table = (*P.DistTables[myTableID]);
    const DistanceTableData &ee_table = (*P.DistTables[0]);
    // get the old value, grad, lapl
    computeU3(P, iat, eI_table.Distances[iat], eI_table.Displacements[iat],
              ee_table.Distances[iat], ee_table.Displacements[iat], Uat[iat],
              dUat[iat], d2Uat[iat], oldUk, olddUk, oldd2Uk);
    if (UpdateMode == ORB_PBYP_RATIO)
    { // ratio-only during the move; need to compute derivatives
      computeU3(P, iat, eI_table.Temp_r.data(), eI_table.Temp_dr,
                ee_table.Temp_r.data(), ee_table.Temp_dr, cur_Uat, cur_dUat,
                cur_d2Uat, newUk, newdUk, newd2Uk);
    }

    for (int jel = 0; jel < Nelec; jel++)
    {
      Uat[jel] += newUk[jel] - oldUk[jel];
      dUat[jel] += newdUk[jel] - olddUk[jel];
      d2Uat[jel] += newd2Uk[jel] - oldd2Uk[jel];
    }

    Uat[iat]   = cur_Uat;
    dUat[iat]  = cur_dUat;
    d2Uat[iat] = cur_d2Uat;

    const int ig = P.GroupID[iat];
    // update compact list elecs_inside
    for (int jat = 0; jat < Nion; jat++)
    {
      bool inside = eI_table.Temp_r[jat] < Ion_cutoff[jat];
      std::vector<int>::iterator iter;
      iter =
          find(elecs_inside(jat, ig).begin(), elecs_inside(jat, ig).end(), iat);
      if (inside)
      {
        if (iter == elecs_inside(jat, ig).end())
          elecs_inside(jat, ig).push_back(iat);
      }
      else
      {
        if (iter != elecs_inside(jat, ig).end())
          elecs_inside(jat, ig).erase(iter);
      }
    }
  }

  inline void recompute(ParticleSet &P)
  {
    const DistanceTableData &eI_table = (*P.DistTables[myTableID]);
    const DistanceTableData &ee_table = (*P.DistTables[0]);

    build_compact_list(P);

    for (int jel = 0; jel < Nelec; ++jel)
    {
      computeU3(P, jel, eI_table.Distances[jel], eI_table.Displacements[jel],
                ee_table.Distances[jel], ee_table.Displacements[jel], Uat[jel],
                dUat[jel], d2Uat[jel], newUk, newdUk, newd2Uk, true);
      // add the contribution from the upper triangle
      for (int kel = 0; kel < jel; kel++)
      {
        Uat[kel] += newUk[kel];
        dUat[kel] += newdUk[kel];
        d2Uat[kel] += newd2Uk[kel];
      }
    }
  }

  inline valT computeU(ParticleSet &P, int jel, const RealType *distjI,
                       const RealType *distjk)
  {
    valT Uj = valT(0);

    const DistanceTableData &eI_table = (*P.DistTables[myTableID]);
    const int jg                      = P.GroupID[jel];

    for (int iat = 0; iat < Nion; ++iat)
      if (distjI[iat] < Ion_cutoff[iat])
      {
        const int ig    = Ions.GroupID[iat];
        const valT r_Ij = distjI[iat];

        for (int kg = 0; kg < eGroups; ++kg)
        {
          const FT &feeI(*F(ig, jg, kg));
          int kel_counter = 0;
          for (int kind = 0; kind < elecs_inside(iat, kg).size(); kind++)
          {
            const int kel = elecs_inside(iat, kg)[kind];
            if (kel != jel)
            {
              DistkI_Compressed[kel_counter] = eI_table.Distances[kel][iat];
              Distjk_Compressed[kel_counter] = distjk[kel];
              kel_counter++;
            }
          }
          Uj += feeI.evaluateV(kel_counter, Distjk_Compressed.data(), r_Ij,
                               DistkI_Compressed.data());
        }
      }
    return Uj;
  }

  inline void computeU3(ParticleSet &P, int jel, const RealType *distjI,
                        const RowContainer &displjI, const RealType *distjk,
                        const RowContainer &displjk, valT &Uj, posT &dUj,
                        valT &d2Uj, vContainer_type &Uk, gContainer_type &dUk,
                        vContainer_type &d2Uk, bool triangle = false)
  {
    constexpr valT czero(0);
    constexpr valT cone(1);
    constexpr valT cminus(-1);
    constexpr valT ctwo(2);
    constexpr valT lapfac = OHMMS_DIM - cone;
    Uj                    = czero;
    dUj                   = posT();
    d2Uj                  = czero;

    const DistanceTableData &eI_table = (*P.DistTables[myTableID]);
    const int jg                      = P.GroupID[jel];

    const int kelmax = triangle ? jel : Nelec;
    std::fill_n(Uk.data(), kelmax, czero);
    std::fill_n(d2Uk.data(), kelmax, czero);
    for (int idim = 0; idim < OHMMS_DIM; ++idim)
      std::fill_n(dUk.data(idim), kelmax, czero);

    valT *restrict val     = mVGL.data(0);
    valT *restrict gradF0  = mVGL.data(1);
    valT *restrict gradF1  = mVGL.data(2);
    valT *restrict gradF2  = mVGL.data(3);
    valT *restrict hessF00 = mVGL.data(4);
    valT *restrict hessF11 = mVGL.data(5);
    valT *restrict hessF22 = mVGL.data(6);
    valT *restrict hessF01 = mVGL.data(7);
    valT *restrict hessF02 = mVGL.data(8);

    for (int iat = 0; iat < Nion; ++iat)
      if (distjI[iat] < Ion_cutoff[iat])
      {
        const int ig       = Ions.GroupID[iat];
        const valT r_Ij    = distjI[iat];
        const posT disp_Ij = cminus * displjI[iat];

        for (int kg = 0; kg < eGroups; ++kg)
        {
          const FT &feeI(*F(ig, jg, kg));
          int kel_counter = 0;
          for (int kind = 0; kind < elecs_inside(iat, kg).size(); kind++)
          {
            const int kel = elecs_inside(iat, kg)[kind];
            if (kel < kelmax && kel != jel)
            {
              DistkI_Compressed[kel_counter] = eI_table.Distances[kel][iat];
              Distjk_Compressed[kel_counter] = distjk[kel];
              DistIndice[kel_counter]        = kel;
              kel_counter++;
            }
          }

          feeI.evaluateVGL(kel_counter, Distjk_Compressed.data(), r_Ij,
                           DistkI_Compressed.data(), val, gradF0, gradF1,
                           gradF2, hessF00, hessF11, hessF22, hessF01, hessF02);

          for (int kel_index = 0; kel_index < kel_counter; kel_index++)
          {
            int kel            = DistIndice[kel_index];
            const posT disp_Ik = cminus * eI_table.Displacements[kel][iat];
            const posT disp_jk = displjk[kel];

            // compute the contribution to jel
            Uj += val[kel_index];
            dUj += gradF0[kel_index] * disp_jk - gradF1[kel_index] * disp_Ij;
            d2Uj -= hessF00[kel_index] + hessF11[kel_index] +
                    lapfac * (gradF0[kel_index] + gradF1[kel_index]) -
                    ctwo * hessF01[kel_index] * dot(disp_jk, disp_Ij);

            // compute the contribution to kel
            Uk[kel] += val[kel_index];
            dUk(kel) = dUk[kel] - gradF0[kel_index] * disp_jk -
                       gradF2[kel_index] * disp_Ik;
            d2Uk[kel] -= hessF00[kel_index] + hessF22[kel_index] +
                         lapfac * (gradF0[kel_index] + gradF2[kel_index]) +
                         ctwo * hessF02[kel_index] * dot(disp_jk, disp_Ik);
          }
        }
      }
  }

  void evaluateGL(ParticleSet &P, ParticleSet::ParticleGradient_t &G,
                  ParticleSet::ParticleLaplacian_t &L, bool fromscratch = false)
  {
    if (fromscratch) recompute(P);
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
}
#endif
